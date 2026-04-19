import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml , plot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 {device=}")

# 1. WSKRZESZAMY WYUCZONĄ FIZYKĘ
model = ml.HardcoreComplexEqualizer().to(device)

# Ładujemy Twój plik z ugotowanymi wagami z RTX'a
model.load_state_dict ( torch.load ( "bpsk_modem_002.pth" , map_location = device , weights_only = True ) )

# BEZWZGLĘDNIE: Blokujemy sieć! (Wyłącza aktualizację wag, włącza tryb analizy)
model.eval ()

# 2. ŁADUJEMY JEDNĄ PACZKĘ Z DYSKU (Twoja krojownia załatwi zrównanie ramki i AGC)
dir_name = Path ( "np.tensors_002_inference" )
lista_X = [ sorted ( dir_name.glob ( "*_rx_samples.npy" ) )[ 0 ] ]
lista_y = [ sorted ( dir_name.glob ( "*_y_train_tensor.pt" ) )[ 0 ] ]

dataset = ml.BPSKDataset ( X_files = lista_X , y_files = lista_y , chunk_samples = ml.CHUNK_SAMPLES_LEN )

# Bierzemy całkowicie losowy kawałek ze środka zbioru
loader = DataLoader ( dataset , batch_size = 128 , shuffle = False )

with torch.no_grad():
    # Wyciągamy wszystko z ładowarki. Żadnych pętli `for`!
    batch_x, batch_y = next(iter(loader))
    
    # Pchamy na kartę i robimy demodulację 
    pred_y = model(batch_x.to(device))
    
    # Spłaszczamy zrzucone paczki od razu w 3 osobne, gigantyczne tasiemce
    sig_raw = batch_x.cpu().numpy().flatten().astype(np.complex128)
    sig_target = batch_y.cpu().numpy().flatten().astype(np.complex128)
    sig_ai = pred_y.cpu().numpy().flatten().astype(np.complex128)

'''
sig_raw = []
sig_ai = []
sig_target = []

# 3. DEMODULACJA AI W LOCIE
print ( "🧠 Przepuszczam sygnał przez Equalizer CVNN..." )
with torch.no_grad () : # Zdejmujemy obciążenie autogradu - sygnał ma tylko przepłynąć
    for batch_x , batch_y in loader :
        pred_y = model ( batch_x.to ( device ) )
    
# Ściągamy tensory do pamięci komputera i zamieniamy na proste Numpy pod Matplotliba
sig_raw.append ( batch_x.cpu ().numpy ().flatten () )
sig_ai.append ( batch_y.cpu ().numpy ().flatten () )
sig_target.append ( pred_y.cpu ().numpy ().flatten () )
sig_raw_all = np.concatenate ( sig_raw )
sig_target_all = np.concatenate ( sig_target )
sig_ai_all = np.concatenate ( sig_ai )
'''

plot.complex_waveform_v0_1_6 ( sig_raw , f"{sig_raw.size=}" )
plot.complex_waveform_v0_1_6 ( sig_target , f"{sig_target.size=}" )
#plot.plot_symbols ( sig_ai , f"{sig_ai.size=}" )

# --- WYKRES 1: SUROWE WEJŚCIE Z RADIA sig_raw
# --- WYKRES 2: ODZYSKANA KONSTELACJA AI sig_ai
# --- WYKRES 3: DOMENA CZASU sig_raw

