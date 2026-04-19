import glob , numpy as np , os , time , tomllib , torch , torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modules import ml

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
    model = ml.HardcoreComplexEqualizer ().to ( device )

    # Ta komenda rozwiązuje problem narzutu języka Python. Silnik Triton w locie
    # zlepi niestandardową zespoloną pętlę LSTM w potężny blok maszynowy.
    #model = torch.compile ( model , mode = "max-autotune" )

    # Nowoczesny wariant Adama (AdamW) z lekkim hamulcem wag dla lepszej stabilności
    optimizer = torch.optim.AdamW ( model.parameters () , lr = ml.LEARNING_RATE , weight_decay = 1e-4 )
    
    # Minimalizacja błędu średniokwadratowego = fizyczny zjazd z EVM
    criterion = nn.MSELoss ()

    EPOCHS = ml.EPOCHS
    print ( "\n🚀 Rozpoczynam trening CVNN (Minimalizacja EVM)..." )
    print ( "🚨 UWAGA: Przy pierwszej przetworzonej paczce ekran CAŁKOWICIE ZAMARZNIE." )
    print ( "🚨 Karta testuje wtedy warianty asemblera w pamięci. Czekaj cierpliwie!\n" )
    
    for epoch in range ( EPOCHS ) :
        model.train ()
        total_loss = 0.0
        
        # Odpalamy stoper dla każdej epoki!
        start_time = time.time ()
        
        for batch_idx , ( batch_x , batch_y ) in enumerate ( loader ) :
            batch_x = batch_x.to ( device )
            batch_y = batch_y.to ( device )
            
            optimizer.zero_grad ()
            
            # Przewidywanie CVNN
            # W TYM MIEJSCU (przy pierwszej iteracji pierwszej epoki) wystąpi zawiecha!
            # TorchInductor właśnie kompiluje "max-autotune".
            predictions = model(batch_x)
            
            # Liczymy fizyczny błąd w kanale radiowym
            #loss = criterion(predictions, batch_y)
            # 🔥 SZYBKI FIX NA BRAK ZESPOLONEGO MSE W CUDA:
            # Rzutujemy płaszczyznę zespoloną na układ 2D (Real, Imag) w locie.
            # Dzięki temu w 100% omijamy braki w bibliotekach NVIDII!
            loss = criterion ( torch.view_as_real ( predictions ) , torch.view_as_real ( batch_y ) )
            
            # Wsteczna propagacja gradientów Wirtingera (magia zespolonego autogradu)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        
        # Oczekiwany rezultat: Epoka 1 potrwa np. 40 sekund, od Epoki 2 czasy pikują na dno!
        print ( f"Epoka [{epoch+1:02d}/{EPOCHS}] | Błąd EVM (MSE): {avg_loss:.5f} | Czas epoki: {epoch_time:.2f} s")
    print ( "\n✅ Trening zakończony! Zespolony potwór został wytrenowany." )
    
    # Zrzucamy wyuczoną fizykę na twardy dysk!
    torch.save ( model.state_dict () , "moj_zespolony_demodulator_SPS4.pth" )
    print ( "💾 Wagi zapisane do pliku. Gotowe do inferencji i testów!" )
    print ( "\n✅ Trening zakończony! Zespolony potwór został wytrenowany." )