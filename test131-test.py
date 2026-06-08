'''
Prompt:
W skrypcie test130-inference.py ładuję sample wejściowe (X_train) jako np.array a docelowy sygnał (y_train) jako tensor. Czy zgodnie ze sztuką pytorch to nie powinno być inference tylko test?
test129-training.py, test130-inference.py, test130-real_inference.py i jeśli trzeba w modules/ml.py, żeby X i y były akceptowane jako np.complex128
'''


import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml , plot

##################################
### SETTINGS #####################
##################################

mode : str = 'test' # 'training' , 'test' lub "inference"
output_decimation : bool = False # Czy zdekodowany sygnał AI ma być zdecymowany (próbkowany co SPS) czy w pełnej rozdzielczości (co próbkę)

##################################
##################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 {device=}")

# 1. WSKRZESZAMY WYUCZONĄ FIZYKĘ
model = ml.HardcoreComplexEqualizer().to(device)

# Ładujemy Twój plik z ugotowanymi wagami z RTX'a
model.load_state_dict ( torch.load ( "bpsk_modem.pth" , map_location = device , weights_only = True ) )

# BEZWZGLĘDNIE: Blokujemy sieć! (Wyłącza aktualizację wag, włącza tryb analizy)
model.eval ()

# 2. ŁADUJEMY GIGANTYCZNY STRUMIEŃ (Brak krojenia, Datasets i DataLoaderów!)
src_dir = Path ( f"pt.{mode}" )
lista_X = sorted ( src_dir.glob ( "*_X_train_samples.npy" ) )
lista_y = sorted ( src_dir.glob ( "*_y_train_tensor.pt" ) )

# Bierzemy pierwsze z brzegu pliki
plik_X = lista_X[0]
plik_y = lista_y[0]

# Ładujemy cały plik prosto do płaskich tablic pamięci RAM
surowy_sygnal = np.load(plik_X).astype(np.complex64)
docelowy_sygnal = torch.load(plik_y, weights_only=True).numpy()

# Ucinamy ogon, aby całkowita długość dzieliła się przez SPS
# (Wymóg sprzętowej decymacji z warstwy Conv1d)
sps = ml.modulation.SPS
reszta = len(surowy_sygnal) % sps
if reszta != 0:
    surowy_sygnal = surowy_sygnal[:-reszta]
    docelowy_sygnal = docelowy_sygnal[:-reszta]

# Transformacja do tensora 3D: [Batch=1, Kanały=1, Czas=CałySygnałZPliku]
# Podwójne unsqueeze(0) zamienia płaską tablicę 1D w tensor pod AI
batch_x = torch.from_numpy(surowy_sygnal).unsqueeze(0).unsqueeze(0).to(device)

# TWARDE AGC
# Zastępuje stare z "BPSKDataset" (które w małej klatce pompowało szum termiczny do jedynki)
batch_x = batch_x / 16384.0

# 3. DEMODULACJA AI W LOCIE
print ( f"🧠 Przepuszczam CAŁY sygnał naraz ({batch_x.size(2)} próbek) przez Equalizer CVNN..." )
with torch.no_grad():
    # Maszyna "połyka" wszystko w jednym przejściu. Faza PLL zostaje idealnie utrzymana!
    pred_y = model(batch_x)
    
# Ściągamy tensory do pamięci komputera i spłaszczamy
sig_raw = surowy_sygnal.astype(np.complex128)
sig_target = docelowy_sygnal.astype(np.complex128)
sig_ai = pred_y.cpu().numpy().flatten().astype(np.complex128)

# --- WYKRESY ---
# UWAGA: Twój model robi teraz piękny zrzut decymacyjny (stride=sps), 
# więc sig_ai jest po wyjściu z sieci SPS-razy krótsze od wejściowego sig_raw!
# Decymujemy z powrotem na sucho również sig_target, by porównać je poprawnie na wykresie:
if output_decimation:
    sig_target_decimated = sig_target[::sps]
else:
    sig_target_decimated = sig_target

plot.complex_waveform_v0_1_6 ( sig_raw , f"Oryginał RAW {sig_raw.size=}" )
plot.complex_waveform_v0_1_6 ( sig_target_decimated , f"Target (zdecymowany) {sig_target_decimated.size=}" )
plot.complex_waveform_v0_1_6 ( sig_ai , f"Odzyskane przez AI {sig_ai.size=}" )