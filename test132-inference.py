import os , torch , numpy as np , time as t
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml, modulation, ops_file, packet , plot

#######################################################################################################################
### SETTINGS ##########################################################################################################
#######################################################################################################################

src : str = 'inference' # 'training' , 'test' lub "inference"
model_version : str = '_v0.1.27' # Wersja modelu do załadowania (np. '' lub '_v0.1.27')

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
    original_len = len ( signal_raw )
    # =========================================================
    # 🔥 OVERLAP-DISCARD (Zakładka 50%) 🔥
    # Rozwiązuje problem "zjedzonych 2 zer" z powodu zimnego startu LSTM!
    # =========================================================
    chunk_size = ml.CHUNK_SAMPLES_LEN  # 8192 sampli = 1 paczka dla GPU
    step = chunk_size // 2             # 4096
    warmup = (chunk_size - step) // 2  # 2048
    # Dopełniamy z przodu i z tyłu o warmup, żeby brzegi sygnału miały rozbieg
    signal_padded = np.pad ( signal_raw , ( warmup , warmup ) , 'constant' )
    # Dopełniamy na końcu, by cała tablica dzieliła się równo na wycinki
    while ( len ( signal_padded ) - chunk_size ) % step != 0 or len ( signal_padded ) < chunk_size :
        signal_padded = np.pad ( signal_padded , ( 0 , 1 ) , 'constant' )
    num_chunks = ( len ( signal_padded ) - chunk_size ) // step + 1
    # Tworzymy paczki z przesunięciem (Overlap)
    windows = np.zeros ( ( num_chunks , chunk_size ) , dtype = np.complex64 )
    for i in range ( num_chunks ) :
        windows[ i ] = signal_padded[ i*step : i*step + chunk_size ]

    # Przebudowa tensora pod sprzętową akcelerację GPU: [Liczba_Paczek, Kanał=1, Czas=8192]. To jest to, co odblokowuje pełną moc RTX!
    # Danie tutaj .to(deviece) od razu przenosi dane do VRAM-u, eliminując wszelkie wąskie gardła PCIe podczas inferencji. Ale moze przy większych plikach trzeba będzie to zrobić partiami, by nie zalać VRAM-u.
    x_tensor = torch.from_numpy ( windows ).unsqueeze ( 1 ).to ( device )
    
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
            # 🔥 WYCINAMY TYLKO ŚRODEK (Odrzucamy brudne krawędzie z zimnego startu!)
            valid_out = predykcja[:, warmup : warmup + step]
            wyniki.append ( valid_out.cpu ().numpy () )

    czas_infer = t.time() - start_infer
    print(f"⏱️ Czas samej sprzętowej inferencji AI: {czas_infer:.3f} s")

    # 5. SPŁASZCZANIE Z POWROTEM DO 1D (Format do Plotly/Dekodera) i SKLEJANIE I ODCINANIE SZTUCZNEGO PADDINGU
    signal_demod = np.concatenate ( wyniki , axis = 0 ).flatten ()
    # Skracamy do oryginalnej długości (ucinamy sztuczny padding)
    signal_demod = signal_demod[:original_len]
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
    
    # =========================================================================
    # 🔥 KROK 2: RADAR BARKERA & AUTO-SYNC (SZUKANIE RAMKI) 🔥
    # =========================================================================
    sps = modulation.SPS
    # Zmiana na znormalizowane twarde punkty BPSK (0 -> -1.0, 1 -> +1.0)
    barker_bipolar = ( packet.BARKER13_BITS.astype ( np.float32 ) * 2.0 ) - 1.0
    best_corr = 0
    best_offset = 0
    best_idx = 0
    is_inverted = False
    print ( "\n🔍 Skanuję odzyskany sygnał radarem korelacyjnym..." )
    # Sprawdzamy wszystkie offsety próbkowania ("oka" symbolu)
    for offset in range ( sps ):
        decimated_soft = samples_demod[ offset : : sps ].real
        # Przesuwamy wzorzec po całym sygnale i mierzymy podobieństwo
        corr = np.correlate ( decimated_soft , barker_bipolar , mode = 'valid' )
        if len ( corr ) == 0: continue
        max_idx = np.argmax(np.abs(corr))
        max_val = np.abs(corr[max_idx])
        # Zapisujemy najlepsze dopasowanie w całym pliku
        if max_val > best_corr:
            best_corr = max_val
            best_offset = offset
            best_idx = max_idx
            is_inverted = (corr[max_idx] < 0)  # Ujemny pik = faza do góry nogami

    print(f"\n🎯 ZNALEZIONO PREAMBUŁĘ BARKER13!")
    print(f"👉 Optymalny offset próbkowania: {best_offset}")
    print(f"👉 Prawdziwy początek ramki (Z racji opóźnienia sieci indeks uległ zmianie!): {best_idx}")
    print(f"👉 Moc korelacji: {best_corr:.2f} / 13.0")
    
    if is_inverted:
        print("⚠️ Sieć zablokowała fazę z obrotem 180° (Typowe w BPSK). Odkręcam konstelację!")

    # ===== KROK 3: SLICER & DECIMATION (TWARDA DECYZJA) =====
    # Zamieniamy miękkie wyjście z sieci na twarde punkty konstelacji BPSK (+1 / -1)
    symbols_decimated = samples_demod[ best_offset : : sps ]
    # Naprawa lustrzanego odbicia konstelacji
    if is_inverted : symbols_decimated = -symbols_decimated
    bpsk_symbols = np.where ( symbols_decimated.real > 0 , 1.0 + 0j , -1.0 + 0j )
    bity = ( symbols_decimated.real > 0 ).astype ( int )
    print(f"\nOdzyskany Barker13 (Oczekiwane {packet.BARKER13_BITS.tolist()}):")
    print(bity[best_idx : best_idx + 13].tolist())
    print("\nKolejne 30 bitów po Barkerze (nagłówek, długość, CRC...):")
    print(bity[best_idx + 13 : best_idx + 43].tolist())
    print("\nFragment zdekodowanego strumienia bitów:")
    print(bity[10035:10043])

plot.complex_waveform_v0_1_6 ( samples_demod , f"after demodulation {timestamp_group}_{samples_demod.size=}" )
plot.complex_waveform_v0_1_6 ( bpsk_symbols , f"after decimation and slicing {timestamp_group}_{bpsk_symbols.size=}" )

if del_src_files :
	for file_path in Path ( src_dir ).glob ( "*" ) :
		if file_path.is_file () : file_path.unlink ( missing_ok = True )