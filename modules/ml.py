import numpy as np
import torch
import torch.nn as nn

from numpy.typing import NDArray
from modules import modulation
from torch.utils.data import Dataset

LEARNING_RATE = 3e-4
EPOCHS = 15

# ========================================================
# NASZ AUTORSKI MODUŁ 1: ZESPOLONA KOMÓRKA BRAMKOWA
# ========================================================
class PureComplexLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Prawdziwa zespolona warstwa liniowa. To tu sieć "zrozumie"
        # obroty Eulera, mieszając osie Real i Imag na krzemie karty graficznej.
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, dtype=torch.complex64)

    def forward(self, x, hx=None):
        # x wchodzi jako pojedyncza próbka w czasie dla całego batcha
        if hx is None:
            # Natywna inicjalizacja pustej pamięci jako zera zespolone (0+0j)
            h = torch.zeros(x.size(0), self.hidden_size, dtype=torch.complex64, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_size, dtype=torch.complex64, device=x.device)
        else:
            h, c = hx

        combined = torch.cat([x, h], dim=1)
        
        # 🔥 Zespolone Mnożenie Macierzy (Tu pracują rdzenie Tensor Twojego RTXa!)
        gates = self.W(combined)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

        # SPLIT-ACTIVATION: 
        # Rozszczepiamy sygnał tylko na ułamek sekundy by przepuścić przez nieliniowe 
        # zawory trygonometryczne (sigmoid/tanh), chroniąc układ przed błędem NaN.
        i = torch.complex(torch.sigmoid(i_gate.real), torch.sigmoid(i_gate.imag))
        f = torch.complex(torch.sigmoid(f_gate.real), torch.sigmoid(f_gate.imag))
        o = torch.complex(torch.sigmoid(o_gate.real), torch.sigmoid(o_gate.imag))
        c_tilde = torch.complex(torch.tanh(c_gate.real), torch.tanh(c_gate.imag))

        # Zespolona aktualizacja stanu fizycznego pamięci
        c_next = f * c + i * c_tilde
        h_next = o * torch.complex(torch.tanh(c_next.real), torch.tanh(c_next.imag))

        return h_next, c_next


# ========================================================
# NASZ AUTORSKI MODUŁ 2: SILNIK CZASOWY (Zastępuje nn.LSTM)
# ========================================================
class PureComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Powołujemy do życia pojedynczą bramkę (zdefiniowaną wyżej)
        self.cell = PureComplexLSTMCell(input_size, hidden_size)

    def forward(self, x):
        # Wejście x: [Batch, Czas (np. 4096), Cechy_Zespolone]
        outputs = []
        hx = None
        
        # Iterujemy po osi czasu kroczek po kroczku.
        # Właśnie po to w głównym skrypcie odpalamy kompilator (max-autotune),
        # żeby sprzętowo zlepił tę pythonową pętlę w ultraszybki asembler!
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], hx)
            hx = (h, c)
            outputs.append(h)
            
        # Zwracamy posklejany, gotowy 3D tensor
        return torch.stack(outputs, dim=1), hx

# ========================================================
# 1. KROJOWNIA DANYCH DLA TWOJEGO y_train (.pt)
# ========================================================
class BPSKDataset ( Dataset ) :
    def __init__(self, X_files: list, y_files: list, chunk_samples: int = 8192):
        
        assert len(X_files) == len(y_files), "BŁĄD: Liczba plików X i Y musi być taka sama!"
        print(f"📡 Ładuję {len(X_files)} par plików sygnałowych z dysku do RAM...")
        
        x_buffers = []
        y_buffers = []
        
        # Pętla ładująca wszystkie pary
        for x_path, y_path in zip(X_files, y_files):
            x_buffers.append(np.load(x_path).astype(np.complex64))
            
            # Parametr weights_only=True dla bezpieczeństwa w nowych wersjach PyTorcha
            y_buffers.append(torch.load(y_path, weights_only=True).to(torch.complex64)) 
            
        # MAGIA WYDAJNOŚCI: Sklejamy wszystko w jeden nieprzerwany ciąg!
        # Dziesiątki Twoich zrzutów z radia stają się nagle jedną potężną osią czasu.
        self.X_raw = np.concatenate(x_buffers, axis=0)
        self.Y_raw = torch.cat(y_buffers, dim=0)
        
        # Żelazna weryfikacja Gęstego Nadzoru (1:1) po spawaniu
        assert len(self.X_raw) == len(self.Y_raw), "FATALNIE: Sklejona oś X i Y mają różną długość!"
        
        self.chunk_samples = chunk_samples
        
        # Całkowita liczba klatek, które wyciśniemy z naszego "węża"
        self.num_chunks = len(self.X_raw) // self.chunk_samples
        print(f"✅ Połączono! Gotowy tasiemiec ma: {len(self.X_raw)} sampli. Klatek do nauki: {self.num_chunks}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_samples
        end = start + self.chunk_samples
        
        # Szybkie cięcie matrycy bez kopiowania fizycznych bitów (Zero-Copy slicing)
        x_np = self.X_raw[start:end]
        X_tensor = torch.from_numpy(x_np).unsqueeze(0) # [1, 8192]
        y_tensor = self.Y_raw[start:end]               # [8192]
        
        # Cyfrowe AGC chroniące wejście modelu przed skokami amplitudy
        max_val = torch.max(torch.abs(X_tensor)) + 1e-9
        X_tensor = X_tensor / max_val
        
        return X_tensor, y_tensor

# ========================================================
# 2. ZESPOLONY POTWÓR BEZ DECYMACJI (FRACTIONALLY SPACED)
# ========================================================
class HardcoreComplexEqualizer ( nn.Module ) :

    def __init__(self):

        super().__init__()
        
        # UWAGA: stride=1 oraz padding='same'.
        # Zrzucamy maski: z 4096 sampli wejściowych wyjdzie równe 4096 
        # sampli po splotach. Brak utraty rozdzielczości czasu!
        self.conv1 = nn.Conv1d ( in_channels = 1 , out_channels = 16 , kernel_size = 7 , stride = 1 , padding = 'same' , dtype = torch.complex64 )
        
        self.lstm = PureComplexLSTM ( input_size = 16 , hidden_size = 64 )
        
        self.fc = nn.Linear ( 64 , 1 , dtype = torch.complex64 )

    def forward ( self , x ) :
        # Wejście x: [Batch, 1, 4096]
        x = self.conv1 ( x )     # Wyjście: [Batch, 16, 4096] (Brak decymacji!)
        x = x.transpose ( 1 , 2 ) # Wyjście: [Batch, 4096, 16]
        
        # Tasiemiec na 4096 kroków wchodzi do LSTM
        x_lstm, _ = self.lstm ( x ) # Wyjście: [Batch, 4096, 64]
        out = self.fc ( x_lstm )    # Wyjście: [Batch, 4096, 1]
        
        # Wypluwamy goły wektor zespolony (zespolony squelch!). 
        return out.squeeze ( -1 )   # Zwraca: [Batch, 4096] (dtype=torch.complex64)

















## Stare śmieci

class AIDemodulator ( nn.Module ) :
    def __init__ ( self , sps = modulation.SPS ) :
        super ().__init__ ()
        # sps (Samples per Symbol) to np. oversampling wpisany do Twojego PlutoSDR
        
        # 1. Ekstrakcja cech. Parametr stride=sps realizuje w locie downsampling!
        self.conv = nn.Conv1d ( in_channels = 2 , out_channels = 16 , kernel_size = sps * 2 , stride = sps , padding = sps//2 )
        self.relu = nn.ReLU ()
        
        # 2. Pamięć fazy (Cyfrowa adaptacyjna "Pętla Costasa")
        self.lstm = nn.LSTM ( input_size = 16 , hidden_size = 32 , batch_first = True )
        
        # 3. Decydent (Zamienia zrekonstruowany symbol na pojedynczą liczbę błędu/prawdopodobieństwa)
        self.classifier = nn.Linear ( 32 , 1 )

    def forward ( self , x ) :
        # x.shape przed splotem: [Batch, 2_Kanały, Długość_w_Próbkach]
        
        x = self.relu ( self.conv ( x ) )
        # Teraz x to już zdecymowane SYMBOLE, a nie próbki
        
        # LSTM potrzebuje kształtu [Batch, Czas_Symboli, Wymiar_Cech], więc obracamy tensor:
        x = x.permute ( 0 , 2 , 1 )
        
        # Oczyszczony ze szumu sygnał przechodzi przez inteligentną odkręcarkę fazy w AI
        x, _ = self.lstm ( x )
        
        # Oceniamy wypluty wynik. Brak Sigmoid pozwala użyć znacznie stabilniejszej funkcji Loss!
        logits = self.classifier ( x )
        
        return logits.squeeze ( -1 ) # Zwraca np. [64, 32_zdekodowane_bity]