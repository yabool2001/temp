import os , torch , numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml, ops_file , plot

#######################################################################################################################
### SETTINGS ##########################################################################################################
#######################################################################################################################

src : str = 'test' # 'training' , 'test' lub "inference"

plt : bool = True # Czy pokazać wykresy z próbkami i wykrytymi ramkami
wrt : bool = True # Czy zapisać y_train_tensor i przyciąć próbki do treningu (wymagane do treningu, ale nie do analizy)
dbg : bool = True

del_src_files : bool = False
del_dst_files : bool = True

#######################################################################################################################
#######################################################################################################################

script_filename = os.path.basename ( __file__ )
np.set_printoptions ( threshold = 10 , edgeitems = 3 ) 

src_dir = Path ( f"pt.{src}" )
dst_dir = Path ( "np.demod" )
Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )

if del_dst_files :
	for file_path in Path ( dst_dir ).glob ( "*" ) :
		if file_path.is_file () :
			file_path.unlink ( missing_ok = True )

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

    # Use first file in the directory whose name is stored in the src_dir variable
    PLIK_RX = sorted ( src_dir.glob ( "*_X_train_samples.npy" ) )[ 0 ]
    if not PLIK_RX.exists () :
        raise FileNotFoundError ( f"Brak plików *_X_train_samples.npy w {src_dir=}" )
    # Extract timestamp group from filename
    timestamp_group = PLIK_RX.stem.split ( "_X_train_samples" , 1)[ 0 ]
    print ( f"{PLIK_RX=}")

    # Magia dzieje się w 1 linijce:
    odzyskane_symbole = demoduluj_na_zywo ( PLIK_RX )
    filename_and_dirname = f"{dst_dir}/{timestamp_group}_demod.npy"
    ops_file.save_complex_samples_2_npf ( filename_and_dirname , odzyskane_symbole )
    
    bity = (odzyskane_symbole.real > 0).astype(int)
    print("\nFragment zdekodowanego strumienia bitów:")
    print(bity[:100])

plot.complex_waveform_v0_1_6 ( odzyskane_symbole , f"{odzyskane_symbole.size=}" )
plot.real_waveform_v0_1_6 ( bity , f"{bity.size=}" )
if del_src_files :
	for file_path in Path ( src_dir ).glob ( "*" ) :
		if file_path.is_file () :
			file_path.unlink ( missing_ok = True )