import numpy as np
from numpy.typing import NDArray
from modules import modulation
import torch
import torch.nn as nn

import numpy as np
import torch
from numpy.typing import NDArray

def iq_to_tensor_v2 (complex_samples: NDArray[np.complex128], seq_len: int = 4096) -> torch.Tensor:
    """ Zamienia rwący wektor complex128 na okna Tensor AI """
    
    # 1. Pocięcie strumienia z radia na równe "klatki"
    num_frames = len(complex_samples) // seq_len
    if num_frames == 0:
        raise ValueError(f"Bufor jest za mały nawet na 1 klatkę (wymaga {seq_len} sampli).")
        
    truncated = complex_samples[: num_frames * seq_len]
    frames = truncated.reshape(num_frames, seq_len)
    
    # 2. Rozbicie na I (Real) oraz Q (Imag) i rzutowanie na float32 pod GPU
    i_chan = np.real(frames).astype(np.float32)
    q_chan = np.imag(frames).astype(np.float32)
    
    # 3. Złożenie w tensor. Kształt: [Ilość_Klatek, 2_Kanały, Czas]
    # Dla Twojego bufora 614k sampli wynik to: [150, 2, 4096]
    iq_tensor = np.stack((i_chan, q_chan), axis=1)
    
    # =========================================================
    # 4. NORMALIZACJA LOKALNA (Cyfrowe AGC per-klatka)
    # =========================================================
    # Szukamy prawdziwej maksymalnej amplitudy zespolonej dla KAŻDEJ z 150 klatek osobno.
    max_mags = np.max(np.abs(frames), axis=1) # Wynik: wektor 1D o dł. 150
    
    # Zmieniamy kształt wektora na [150, 1, 1], aby matematyka Pythona wiedziała,
    # jak bezpiecznie rozciągnąć to i podzielić macierz 3D (tzw. Broadcasting).
    max_mags = max_mags.reshape(num_frames, 1, 1)
    
    # Dzielenie. Klatka nr 5 dzieli się TYLKO przez max z klatki nr 5.
    # Zabezpieczenie 1e-9 ratuje przed dzieleniem przez zero przy totalnej ciszy na wejściu.
    iq_tensor = iq_tensor / (max_mags + 1e-9)
    
    tensor_out = torch.from_numpy(iq_tensor)
    
    # --- UWAGA ARCHITEKTONICZNA ---
    # Jeśli wchodzisz z tym od razu do nn.LSTM (nie masz na początku nn.Conv1d), 
    # musisz odkomentować poniższą linijkę, by przesunąć czas do środka:
    # tensor_out = tensor_out.transpose(1, 2)  # Zmienia kształt na [150, 4096, 2]
    
    return tensor_out

def iq_to_tensor_v1 ( complex_samples : NDArray[ np.complex128 ] , seq_len: int = 256 ) -> torch.Tensor :
    """ Zamienia rwący wektor complex128 na okna Tensor AI: [Batch_Size, 2_Kanały, Długość_Ramki] """
    # 1. Pocięcie strumienia z radia na równe "klatki"
    num_frames = len ( complex_samples ) // seq_len
    truncated = complex_samples[ : num_frames * seq_len ] # Ucinamy resztkę
    frames = truncated.reshape ( num_frames , seq_len )
    
    # 2. Rozbicie na I (Real) oraz Q (Imag) i rzutowanie na wymuszone float32
    i_chan = np.real ( frames ).astype ( np.float32 )
    q_chan = np.imag ( frames ).astype ( np.float32 )
    
    # 3. Złożenie w tensor [Ilość_Klatek, 2_Kanały, Okno_Czasowe]
    iq_tensor = np.stack ( (i_chan , q_chan ) , axis = 1 )
    
    # 4. Normalizacja (Krytyczne! AI nienawidzi dużych liczb, oczekuje ich w okolicach [-1.0, 1.0])
    # Używamy małej stałej 1e-9 by uniknąć dzielenia przez zero przy pustym szumie
    iq_tensor = iq_tensor / ( np.max ( np.abs ( iq_tensor ) ) + 1e-9 )
    
    return torch.from_numpy ( iq_tensor )

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