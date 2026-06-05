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
from numpy.typing import NDArray
from modules import filters , ops_file, packet , plot

script_filename = os.path.basename ( __file__ )
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

#######################################################################################################################
### SETTINGS ##########################################################################################################
#######################################################################################################################

plt : bool = True # Czy pokazać wykresy z próbkami i wykrytymi ramkami
wrt : bool = True # Czy zapisać y_train_tensor i przyciąć próbki do treningu (wymagane do treningu, ale nie do analizy)
dbg : bool = True
del_pt_files : bool = True
del_np_files : bool = False
trn : bool = False

#######################################################################################################################
#######################################################################################################################

samples_dir = Path ( "np.samples" )

if trn :
	tensors_dir = Path ( "pt.training" )
	clp : bool = True # Czy przyciąć próbki do długości ramki (wymagane do treningu, ale nie do analizy)
else :
	tensors_dir = Path ( "np.inference" )
	clp : bool = False # Czy przyciąć próbki do długości ramki (wymagane do treningu, ale nie do analizy)

if del_pt_files :
	for file_path in Path ( tensors_dir ).glob ( "*" ) :
		if file_path.is_file () :
			file_path.unlink ( missing_ok = True )

device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
print ( f"torch {device=}" )


# Znajdź unikalne grupy plików na podstawie timestampu w nazwie dla plików *_rx_samples_*.npy które nie były jeszcze obrobione!!!
timestamp_groups = sorted ( { p.name.split ( "_rx_samples_" , 1 )[ 0 ] for p in samples_dir.glob("*_rx_samples_*.npy") } )

# Główna pętla przetwarzająca każdą grupę plików 
for timestamp_group in timestamp_groups :

	samples_files = sorted ( samples_dir.glob ( f"{timestamp_group}_rx_samples_*.npy" ) )
	timestamps = sorted ( { p.stem.split(f"{timestamp_group}_rx_samples_", 1)[1] for p in samples_dir.glob(f"{timestamp_group}_rx_samples_*.npy") } )
	if dbg : print ( f"\n{timestamp_group=} : {timestamps=}" )
	if not samples_files :
		raise FileNotFoundError ( f"Brak plikow {timestamp_group}_rx_samples_*.npy w katalogu {samples_dir}" )
	rx_samples = packet.RxSamples ()
	
	# Mała pętla agregująca wszystkie sample z plików należących do tej samej grupy (czyli mających ten sam timestamp w nazwie) do obiektu rx_samples.
	for samples_file in samples_files :
		rx_samples.rx ( file_name = str ( samples_file ) , concatenate = True )
		print ( f"{rx_samples.concatenates=}" )
	rx_samples.detect_frames ( deep = False , filter = True , correct = False , add_peak_at_0 = False )
	#rx_samples_frames_first_sample_idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_first_sample_idx for frame in rx_samples.frames ] , dtype = np.uint32 )
	rx_samples.tx_active_symbols = ops_file.open_samples_from_npf ( filename = f"{samples_dir.name}/{timestamp_group}_tx_active_symbols.npy" )
	rx_samples.tx_samples = ops_file.open_samples_from_npf ( filename = f"{samples_dir.name}/{timestamp_group}_tx_samples.npy" )
	my_idx1 : NDArray [ np.uint32 ] = np.array ( [ rx_samples.frames[0].frame_start_abs_first_sample_idx , rx_samples.frames[0].frame_start_abs_first_sample_idx + rx_samples.tx_active_symbols.size + filters.ADD_SAMPLES_TAIL_OFFSET ] , dtype = np.uint32 )
	if plt :
		#rx_samples.plot_samples ( title = f"{script_filename} {timestamp_group} concatenated rx_samples | " , mark_first_active_samples = True )
		plot.complex_waveform_v0_1_6 ( rx_samples.samples , f"{script_filename} {timestamp_group} concatenated rx_samples | {rx_samples.samples.size=}" , marker_peaks = my_idx1 )
		#rx_samples.plot_symbols ( rx_samples.tx_active_symbols , title = f"{script_filename} {timestamp_group} tx_active_symbols" )

	# Szukanie dopasowania nagłówka ramki zaczynając od rx
	first_active_rx_sample_idx : np.uint32 = None
	if rx_samples.frames is not None and len ( rx_samples.frames ) > 0 :
		tx_samples = packet.RxSamples ()
		tx_samples.rx ( file_name = str ( f"{samples_dir.name}/{timestamp_group}_tx_samples.npy" ) )
		tx_samples.detect_frames ( deep = False , filter = True , correct = False , add_peak_at_0 = True )
		for rx_frame in rx_samples.frames :
			for tx_frame in tx_samples.frames :
				if dbg :
					print ( f"rx: {rx_frame.packet_len}	{packet.pad_bits2bytes ( rx_frame.header_bits )}	{rx_frame.frame_start_abs_first_sample_idx}" )
					print ( f"tx: {tx_frame.packet_len}	{packet.pad_bits2bytes ( tx_frame.header_bits )}	{tx_frame.frame_start_abs_first_sample_idx}" )
				if np.array_equal ( rx_frame.header_bits , tx_frame.header_bits ) :
					first_active_rx_sample_idx = rx_frame.frame_start_abs_first_sample_idx - tx_frame.frame_start_abs_first_sample_idx + filters.PEAK_TO_ACTIVE_SAMPLE_OFFSET
					#first_active_rx_sample_idx = rx_frame.frame_start_abs_first_sample_idx
					if dbg : print ( f"\r\nZnaleziono dopasowanie ramki: {timestamp_group=} {first_active_rx_sample_idx=}" )
					my_idx2 = np.array ( [ first_active_rx_sample_idx , first_active_rx_sample_idx + rx_samples.tx_active_symbols.size ] , dtype = np.uint32 )
					break
			if first_active_rx_sample_idx is not None :
				rx_samples.create_tx_symbols ( first_active_symbols_idx = first_active_rx_sample_idx )
				#if plt : rx_samples.plot_symbols ( rx_samples.tx_symbols , title = f"{script_filename} {timestamp_group} tx_symbols" )
				break

	if first_active_rx_sample_idx is not None :
		rx_samples.clip_samples_and_create_tensor_4_training_only_active_symbols ( first_idx = first_active_rx_sample_idx )
		if plt : plot.complex_waveform_v0_1_6 ( rx_samples.X_train_samples , title = f"{script_filename} {timestamp_group} X_train_samples {rx_samples.X_train_samples.size=}" , marker_peaks = my_idx2 )
		if plt : plot.flat_tensor_v0_1_18 ( rx_samples.y_train_tensor , title = f"{script_filename} {timestamp_group} y_train_tensor" , marker_idx = first_active_rx_sample_idx )
		if wrt : rx_samples.save_train_data ( timestamp_group = f"{timestamp_group}" , dir_name = tensors_dir.name , add_timestamp = False )
		if del_np_files :
			for file_path in Path ( samples_dir ).glob ( f"{timestamp_group}_*.*" ) :
				if file_path.is_file () :
					file_path.unlink ( missing_ok = True )
	else :
		print ( f"Nie znaleziono dopasowania ramki: {timestamp_group}" )
	if dbg : print ( f"Zakończono przetwarzanie grupy: {timestamp_group}" )