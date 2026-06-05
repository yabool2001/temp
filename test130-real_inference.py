import os , torch , numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml , plot

script_filename = os.path.basename ( __file__ )
np.set_printoptions ( threshold = 10 , edgeitems = 3 ) 

device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
print ( f"🔥 {device=}" )

def demoduluj_na_zywo ( sciezka_do_pliku_npy: str ) -> np.ndarray :
    """
    Czysta funkcja produkcyjna. Zero krojowni i datasetów z PyTorcha.
    Wchodzi szum -> Wychodzi zdemodulowany sygnał. 
    ROZWIĄZANIE 1: Zamiast cięcia w okna podajemy GIGANTYCZNY strumień (brak resetu fazy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inicjalizacja produkcyjnego silnika AI na: {device}")

    # 1. WSKRZESZAMY MÓZG
    model = ml.HardcoreComplexEqualizer().to(device)
    model.load_state_dict ( torch.load ( "bpsk_modem.pth" , map_location = device , weights_only = True ) )
    
    # KRYTYCZNE: Blokujemy sieć do odczytu (tryb ewaluacji). 
    model.eval()

    # 2. NASŁUCH Z ANTENY (Czytamy cały plik wejściowy X!)
    print(f"📡 Wczytuję surowy eter z: {sciezka_do_pliku_npy}")
    surowy_sygnal = np.load(sciezka_do_pliku_npy).astype(np.complex64)
    
    # 3. KROJENIE W LOCIE USUNIĘTE.
    # Ucinamy tylko ogon, by sygnał wpasował się równo w filtry Conv1d bez wyrzucania błędu.
    sps = ml.modulation.SPS
    reszta = len(surowy_sygnal) % sps
    if reszta != 0:
        surowy_sygnal = surowy_sygnal[:-reszta]
    
    # TWORZYMY POTĘŻNĄ MATRYCĘ: [Batch=1, Kanał=1, Czas=CałyPlik]
    x_tensor = torch.from_numpy(surowy_sygnal).unsqueeze(0).unsqueeze(0).to(device)
    
    # SPRZĘTOWE AGC 
    x_tensor = x_tensor / 16384.0

    # 4. 🔥 STRZAŁ Z AI 🔥
    print(f"🧠 RTX 5080 połyka wszystkie {x_tensor.size(2)} sampli naraz w jednym strumieniu...")
    with torch.no_grad():
        predykcja_ai = model(x_tensor) 

    # 5. SPŁASZCZANIE Z POWROTEM DO 1D (Format do Plotly/Dekodera)
    zdemodulowany_sygnal = predykcja_ai.cpu().numpy().flatten().astype(np.complex128)
    
    print(f"✅ Demodulacja zakończona! Zwrócono zdecymowany strumień długości {zdemodulowany_sygnal.size}")
    return zdemodulowany_sygnal

# ==========================================
# UŻYCIE W TWOIM PROGRAMIE GŁÓWNYM
# ==========================================
if __name__ == "__main__":
    
    # Bierzemy plik z eteru
    PLIK_RX = "np.inference/1780580221043_X_train_samples.npy"

    # Magia dzieje się w 1 linijce:
    odzyskane_symbole = demoduluj_na_zywo ( PLIK_RX )
    
    bity = (odzyskane_symbole.real > 0).astype(int)
    print("\nFragment zdekodowanego strumienia bitów:")
    print(bity[:100])

plot.complex_waveform_v0_1_6 ( odzyskane_symbole , f"{odzyskane_symbole.size=}" )
plot.real_waveform_v0_1_6 ( bity , f"{bity.size=}" )