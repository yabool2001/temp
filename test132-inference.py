import os , torch , numpy as np , time as t
from numpy.typing import NDArray
from pathlib import Path
from modules import ml, modulation, ops_file, packet , plot

################################################################################
### SETTINGS ###################################################################
src : str = 'inference'
model_version : str = '_v0.1.31' 
plt : bool = True 
dbg : bool = True
wrt : bool = True
del_src_files : bool = False
del_dst_files : bool = True
################################################################################
################################################################################

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

    model = ml.HardcoreComplexEqualizer ().to ( device )
    model.load_state_dict ( torch.load ( f"bpsk_modem{model_version}.pth" , map_location = device , weights_only = True ) )
    model.eval ()

    print ( f"📡 Wczytuję strumień z {filename_and_dirname=}")
    signal_raw = np.load ( filename_and_dirname ).astype ( np.complex64 )
    original_len = len(signal_raw)
    
    # =========================================================
    # 🔥 OVERLAP-DISCARD (Zakładka 50%) 🔥
    # Rozwiązuje problem "zjedzonych 2 zer" z powodu zimnego startu LSTM!
    # =========================================================
    chunk_size = ml.CHUNK_SAMPLES_LEN  # 8192
    step = chunk_size // 2             # 4096
    warmup = (chunk_size - step) // 2  # 2048

    # Dopełniamy z przodu i z tyłu o warmup, żeby brzegi sygnału miały rozbieg
    signal_padded = np.pad(signal_raw, (warmup, warmup), 'constant')
    
    # Dopełniamy na końcu, by cała tablica dzieliła się równo na wycinki
    while (len(signal_padded) - chunk_size) % step != 0 or len(signal_padded) < chunk_size:
        signal_padded = np.pad(signal_padded, (0, 1), 'constant')

    num_chunks = (len(signal_padded) - chunk_size) // step + 1

    # Tworzymy paczki z przesunięciem (Overlap)
    windows = np.zeros((num_chunks, chunk_size), dtype=np.complex64)
    for i in range(num_chunks):
        windows[i] = signal_padded[i*step : i*step + chunk_size]

    x_tensor = torch.from_numpy(windows).unsqueeze(1).to(device)

    # ZŁOTE AGC
    max_vals = torch.max ( torch.abs ( x_tensor ) , dim = 2 , keepdim = True )[ 0 ] + 1e-9
    x_tensor = x_tensor / max_vals

    BATCH_SIZE = 64
    wyniki = []
    
    print(f"🧠 RTX 5080 połyka {num_chunks=} nachodzących na siebie bloków...")
    start_infer = t.time()

    with torch.no_grad () :
        for i in range ( 0 , num_chunks , BATCH_SIZE ) :
            batch = x_tensor[ i : i + BATCH_SIZE ]
            predykcja = model ( batch )
            
            # 🔥 WYCINAMY TYLKO ŚRODEK (Odrzucamy brudne krawędzie z zimnego startu!)
            valid_out = predykcja[:, warmup : warmup + step]
            wyniki.append ( valid_out.cpu ().numpy () )

    czas_infer = t.time() - start_infer
    print(f"⏱️ Czas samej sprzętowej inferencji AI: {czas_infer:.3f} s")

    signal_demod = np.concatenate ( wyniki , axis = 0 ).flatten ()
    
    # Skracamy do oryginalnej długości (ucinamy sztuczny padding)
    signal_demod = signal_demod[:original_len]
    
    print(f"✅ Demodulacja zakończona bezszwowo! Zwrócono strumień długości {signal_demod.size}")    
    return signal_demod.astype ( np.complex64 )


# ==========================================
# UŻYCIE W TWOIM PROGRAMIE GŁÓWNYM
# ==========================================
if __name__ == "__main__":

    rx_filename_and_dirname = sorted ( src_dir.glob ( "*_X_train_samples*.npy" ) )[ 0 ]
    if not rx_filename_and_dirname.exists () :
        raise FileNotFoundError ( f"Brak plików *_X_train_samples.npy w {src_dir=}" )
        
    timestamp_group = rx_filename_and_dirname.stem.split ( "_X_train_samples" , 1 )[ 0 ]
    print ( f"{rx_filename_and_dirname=}")
    
    # =========================================================================
    # KROK 1: MIĘKKA DEMODULACJA AI
    # =========================================================================
    ai_demod_samples = demod ( str ( rx_filename_and_dirname ) )
    rx_ai_samples = packet.RxSamples ()
    rx_ai_samples.rx ( complex_samples = ai_demod_samples , concatenate = False )
    rx_ai_samples.detect_frames ( deep = False , samples_filtered = True , correct_samples = False , add_peak_at_0 = False )
    
    if plt:
        plot.complex_waveform_v0_1_6 ( ai_demod_samples , f"{script_filename} {timestamp_group} AI samples {ai_demod_samples.size=}" )
        #plot.real_waveform_v0_1_6 ( ai_symbols.real[ first_symbol_abs_idx : last_symbol_abs_idx ] , f"{script_filename} AI symbols.real {timestamp_group} {ai_symbols.size=}" )
        #plot.real_waveform_v0_1_6 ( y_train_symbols.real[ first_symbol_abs_idx : last_symbol_abs_idx ] , f"{script_filename} y_train symbols.real {timestamp_group} {y_train_symbols.size=}" )

    if wrt :
        ops_file.save_complex_samples_2_npf ( f"{dst_dir}/{timestamp_group}_ai_demod_samples.npy" , ai_demod_samples )


    if del_src_files :
        for file_path in Path ( src_dir ).glob ( "*" ) :
            if file_path.is_file () : file_path.unlink ( missing_ok = True )
