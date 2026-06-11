'''
Odczytanie próbek z plików npy:
{timestamp_group}_rx_samples_{timestamp}.npy - surowe próbki rx
{timestamp_group}_tx_active_symbols.npy - aktywne próbki wyciągnięte ze składowej real sygnału samples_4_pluto (samples_4_pluto.real) i zaokrąglone do wartości -1+j0 i 1+j0. Wartości obejmują tylo aktywne symbole TX, zaczynające się od  (bez rozbiegówki i wygaszenia, bez 0+j0)
Może w przyszłości {timestamp_group}_tx_samples.npy wykorzystam dodatkowo do wielkiej korelacji całego przebiegu ale na razie wydaje się to niewykonalne ze względu na rozmiar danych i czas potrzebny na korelację.

Agregowanie sampli raw do obiektu RxSamples, wykrywanie ramek, dopasowywanie ich do ramek z próbek tx,

Tworzenie tensorów y_train i przycinanie ich razem z samplami i zapisywanie ich do plików w katalogu wrt_dir:
- {timestamp_group}_X_train_samples.npy - wejsciowe raw sample przycięte ale bez filtrowania i korekcji
- {timestamp_group}_y_train_tensor.pt - tensor z symbolami tx przycięty do tej samej długości co X_train_samples, gotowy do treningu modelu ML
'''

# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

# dobra grupa do testów: 1777232234209_rx_samples_1777232234490.npy w np.tensors_005

import numpy as np , os , tomllib , torch
from pathlib import Path
from modules import packet

script_filename = os.path.basename ( __file__ )
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

#######################################################################################################################
### SETTINGS ##########################################################################################################
#######################################################################################################################

mode : str = 'training' # 'training' , 'test' lub "inference"
y_train_tensor_src : str = 'active_samples' # 'symbols': do tworzenia X_train_samples używamy symboli tx (czyli próbek z pliku {timestamp_group}_tx_active_symbols.npy),
									# 'active_samples': do tworzenia X_train_samples używamy surowych próbek rx (czyli próbek z pliku {timestamp_group}_rx_samples_{timestamp}.npy)
									# ale tylko tych które odpowiadają aktywnym symbolom tx, czyli tych które są w ramce i pozycjach odpowiadających symbolom tx.
samples_filtered_4_X_train_samples : bool = False # czy do tworzenia X_train_samples używać surowych próbek (samples_raw) czy próbek po filtracji (samples_filtered)
X_y_clipping_mode : str = 'symbols_only' # 'balanced': przycinamy próbki do długości ramki, ale dodajemy trochę rozbiegówki i wygaszenia,
									# 'symbols_only': przycinamy dokładnie do długości ramki bez rozbiegówki i wygaszenia, 


plt : bool = False # Czy pokazać wykresy z próbkami i wykrytymi ramkami
wrt : bool = True # Czy zapisać y_train_tensor i przyciąć próbki do treningu (wymagane do treningu, ale nie do analizy)
dbg : bool = True

del_src_files : bool = False
del_dst_files : bool = True

#######################################################################################################################
#######################################################################################################################

src_dir = Path ( f"np.{mode}" )
dst_dir = Path ( f"pt.{mode}" )
Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )

if mode in ('training', 'test', 'inference') :
    if dbg : print ( f"Running in {mode=}." )
else :
    raise ValueError ( f"Unknown {mode=}. Available modes: 'training', 'test' or 'inference'." )

if del_dst_files :
	for file_path in Path ( dst_dir ).glob ( "*" ) :
		if file_path.is_file () :
			file_path.unlink ( missing_ok = True )

device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
if dbg : print ( f"torch {device=}" )

# Znajdź unikalne grupy plików na podstawie timestampu w nazwie dla plików *_rx_samples_*.npy które nie były jeszcze obrobione!!!
timestamp_groups = sorted ( { p.name.split ( "_rx_samples_" , 1 )[ 0 ] for p in src_dir.glob("*_rx_samples_*.npy") } )

# Główna pętla przetwarzająca każdą grupę plików 
for timestamp_group in timestamp_groups :

	samples_files = sorted ( src_dir.glob ( f"{timestamp_group}_rx_samples_*.npy" ) )
	timestamps = sorted ( { p.stem.split(f"{timestamp_group}_rx_samples_", 1)[1] for p in src_dir.glob(f"{timestamp_group}_rx_samples_*.npy") } )
	if dbg : print ( f"\n{timestamp_group=} : {timestamps=}" )
	if not samples_files :
		raise FileNotFoundError ( f"Brak plikow {timestamp_group}_rx_samples_*.npy w katalogu {src_dir}" )
	rx_samples = packet.RxSamples ()
	
	# Mała pętla agregująca wszystkie sample z plików należących do tej samej grupy (czyli mających ten sam timestamp w nazwie) do obiektu rx_samples.
	for samples_file in samples_files :
		rx_samples.rx ( file_name = str ( samples_file ) , concatenate = True )
		if dbg : print ( f"{rx_samples.concatenates=}" )
	rx_samples.detect_frames ( deep = False , samples_filtered = True , correct_samples = False , add_peak_at_0 = False )
	#if plt : rx_samples.plot_samples ( title = f"{script_filename} {timestamp_group} concatenated rx_samples " , samples_filtered = False , mark_samples = True )
	first_symbol_idx = rx_samples.create_X_train_samples_and_y_train_tensor ( src_dir = src_dir , timestamp_group = timestamp_group , X_train_samples_filtered = samples_filtered_4_X_train_samples , symbols_src = y_train_tensor_src )
	no_X_train_samples_created : int = 0
	if first_symbol_idx is not None :
		no_X_train_samples_created += 1
		if plt : rx_samples.plot_X_and_y ( title = f"{script_filename} {timestamp_group} X_train_samples and y_train_tensor before clipping" , mark_samples = True )
		rx_samples.clip_X_train_samples_and_y_train_tensor ( clipping_mode = X_y_clipping_mode )
		if plt : rx_samples.plot_X_and_y ( title = f"{script_filename} {timestamp_group} X_train_samples and y_train_tensor after clipping" , mark_samples = False )
		if wrt : rx_samples.save_train_data ( timestamp_group = f"{timestamp_group}" , dir_name = dst_dir.name , add_timestamp = False )
		if del_src_files :
			for file_path in Path ( src_dir ).glob ( f"{timestamp_group}_*.*" ) :
				if file_path.is_file () :
					file_path.unlink ( missing_ok = True )
	else :
		print ( f"Nie znaleziono dopasowania ramki: {timestamp_group}" )
	if dbg : print ( f"Zakończono przetwarzanie grupy: {timestamp_group} {no_X_train_samples_created=}" )