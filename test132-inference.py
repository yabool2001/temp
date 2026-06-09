import os , torch , numpy as np , time as t
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml, ops_file , plot

#######################################################################################################################
### SETTINGS ##########################################################################################################
#######################################################################################################################

src : str = 'inference' # 'training' , 'test' lub "inference"
model_version : str = '' # Wersja modelu do załadowania (np. '' lub '_v0.1.27')

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
#Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )
dst_dir.mkdir ( parents = True , exist_ok = True )

if del_dst_files :
	for file_path in Path ( dst_dir ).glob ( "*" ) :
		if file_path.is_file () : file_path.unlink ( missing_ok = True )

device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
print ( f"🔥 {device=}" )

def demod ( filename_and_dirname : str ) -> np.ndarray :

    # 1: WSKRZESZAMY MÓZG (Tryb Ewaluacji zabezpiecza wagi! Blokujemy sieć tylko do odczytu)
    model = ml.HardcoreComplexEqualizer ().to ( device )
    model.load_state_dict ( torch.load ( f"bpsk_modem{model_version}.pth" , map_location = device , weights_only = True ) )
    model.eval ()

    # 2. Wczytujemy cały plik wejściowy i dopełniamy zerami (PADDING), by plik dzielił się równo przez ml.CHUNK_SAMPLES_LEN (8192)
    print ( f"📡 Wczytuję strumień z {filename_and_dirname=}")
    signal_raw = np.load ( filename_and_dirname ).astype ( np.complex64 )
    chunk_size = ml.CHUNK_SAMPLES_LEN # np. 8192 sampli = 1 paczka dla GPU
    reszta = len ( signal_raw ) % chunk_size
    pad_len = 0
    if reszta != 0:
        pad_len = chunk_size - reszta
        signal_raw = np.pad ( signal_raw , ( 0 , pad_len ) , 'constant' )
    num_chunks = len ( signal_raw ) // chunk_size

    # Przebudowa tensora pod sprzętową akcelerację GPU: [Liczba_Paczek, Kanał=1, Czas=8192]. To jest to, co odblokowuje pełną moc RTX!
    # Danie tutaj .to(deviece) od razu przenosi dane do VRAM-u, eliminując wszelkie wąskie gardła PCIe podczas inferencji. Ale moze przy większych plikach trzeba będzie to zrobić partiami, by nie zalać VRAM-u.
    x_tensor = torch.from_numpy ( signal_raw ).view ( num_chunks , 1 , chunk_size ).to ( device )
    
    # 3. 🔥 ZŁOTE AGC 🔥
    # Normalizujemy KAŻDĄ paczkę 8192 osobno, DOKŁADNIE tak, jak w treningu!
    max_vals = torch.max ( torch.abs ( x_tensor ) , dim = 2 , keepdim = True )[ 0 ] + 1e-9
    x_tensor = x_tensor / max_vals

    BATCH_SIZE = 64
    wyniki = []
    
    print(f"🧠 RTX 5080 połyka {num_chunks=} bloków sygnału. Trwa inferencja...")
    start_infer = t.time()

    # 4. 🔥 STRZAŁ Z AI 🔥
    with torch.no_grad () :
        for i in range ( 0 , num_chunks , BATCH_SIZE ) :
            batch = x_tensor[ i : i + BATCH_SIZE ]
            predykcja = model ( batch )
            wyniki.append ( predykcja.cpu ().numpy () )

    czas_infer = t.time() - start_infer
    print(f"⏱️ Czas samej sprzętowej inferencji AI: {czas_infer:.3f} s")

    # 5. SPŁASZCZANIE Z POWROTEM DO 1D (Format do Plotly/Dekodera) i SKLEJANIE I ODCINANIE SZTUCZNEGO PADDINGU
    signal_demod = np.concatenate ( wyniki , axis = 0 ).flatten ()
    if pad_len != 0:
        signal_demod = signal_demod[ : -pad_len ]
    print(f"✅ Demodulacja zakończona! Zwrócono zdecymowany strumień długości {signal_demod.size}")    
    return signal_demod.astype ( np.complex128 )

# ==========================================
# UŻYCIE W TWOIM PROGRAMIE GŁÓWNYM
# ==========================================
if __name__ == "__main__":

    # Use first file in the directory whose name is stored in the src_dir variable
    rx_filename_and_dirname = sorted ( src_dir.glob ( "*_X_train_samples.npy" ) )[ 0 ]
    if not rx_filename_and_dirname.exists () :
        raise FileNotFoundError ( f"Brak plików *_X_train_samples.npy w {src_dir=}" )
    # Extract timestamp group from filename
    timestamp_group = rx_filename_and_dirname.stem.split ( "_X_train_samples" , 1)[ 0 ]
    print ( f"{rx_filename_and_dirname=}")

    # Magia dzieje się w 1 linijce:
    samples_demod = demod ( filename_and_dirname = rx_filename_and_dirname )
    
    filename_and_dirname = f"{dst_dir}/{timestamp_group}_demod.npy"
    ops_file.save_complex_samples_2_npf ( filename_and_dirname , samples_demod )
    
    # ===== KROK 3: SLICER (TWARDA DECYZJA) =====
    # Zamieniamy miękkie wyjście z sieci na twarde punkty konstelacji BPSK (+1 / -1)
    bpsk_symbols = np.where(samples_demod.real > 0, 1.0 + 0j, -1.0 + 0j)

    bity = (samples_demod.real > 0).astype(int)
    print("\nFragment zdekodowanego strumienia bitów:")
    print(bity[:13])

plot.complex_waveform_v0_1_6 ( samples_demod , f"{timestamp_group}_{samples_demod.size=}" )
plot.complex_waveform_v0_1_6 ( bpsk_symbols , f"{timestamp_group}_{bpsk_symbols.size=}" )

if del_src_files :
	for file_path in Path ( src_dir ).glob ( "*" ) :
		if file_path.is_file () : file_path.unlink ( missing_ok = True )