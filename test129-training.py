import glob , numpy as np , os , time , tomllib , torch , torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modules import ml , modulation , packet

script_filename = os.path.basename ( __file__ )
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

tensors_dir = Path ( "np.tensors" )

if __name__ == "__main__":
    
    # 1. Odblokowanie sprzętowych dopalaczy w układach RTX (TF32)
    # Bezwzględny "must-have" przy operacjach macierzowych na najnowszych GPU
    torch.set_float32_matmul_precision ( 'high' )
    
    # Skoro pod maską siedzi potwór, to nie zadajemy pytań - lecimy w CUDA
    device = torch.device ( "cuda" )
    print(f"\n🔥{device=} {torch.__version__=} {torch.cuda.get_device_name ( 0 )=}")

    # 2. Ładowanie danych
    # Dataset w locie zrzutuje Twoje pliki .npy z uciążliwych 128-bitowych
    # na natywne 64-bitowe dla najwyższej przepustowości VRAM.
    lista_plikow_X = sorted ( tensors_dir.glob ( "*_rx_samples.npy") )
    lista_plikow_y = sorted ( tensors_dir.glob ( "*_y_train_tensor.pt") )
    if not lista_plikow_X:
        raise RuntimeError ( "Nie znaleziono żadnych plików .npy w katalogu!" )
    if not lista_plikow_y:
        raise RuntimeError ( "Nie znaleziono żadnych plików .pt w katalogu!" )
    print("📡 Ładowanie zrzutów z PlutoSDR z dysku...")
    dataset = ml.BPSKDataset (
        X_files = lista_plikow_X ,
        y_files = lista_plikow_y ,
        chunk_samples = 4096 )

    # RTX 5080 ma ocean ultraszybkiego VRAM-u (16GB), ładujemy solidny Batch Size. 
    # pin_memory=True i num_workers wymuszają bezpośredni transfer DMA po PCIe!
    loader = DataLoader ( dataset ,
        batch_size = 64 ,
        shuffle = True ,
        pin_memory = True ,
        num_workers = 4 )
    
    # 3. Powołanie do życia naszej w 100% natywnej, zespolonej architektury
    model = packet.HardcoreComplexEqualizer ().to ( device )

    # Ta komenda rozwiązuje problem narzutu języka Python. Silnik Triton w locie
    # zlepi niestandardową zespoloną pętlę LSTM w potężny blok maszynowy.
    model = torch.compile ( model , mode = "max-autotune" )