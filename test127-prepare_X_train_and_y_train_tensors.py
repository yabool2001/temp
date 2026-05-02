# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

# dobra grupa do testów: 1777232234209_rx_samples_1777232234490.npy w np.tensors_005

import numpy as np , os , tomllib , torch
from pathlib import Path
from numpy.typing import NDArray
from modules import filters , modulation, ops_file, packet , plot

script_filename = os.path.basename ( __file__ )
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

clp : bool = False # Czy przyciąć próbki do długości ramki (wymagane do treningu, ale nie do analizy)
plt : bool = True # Czy pokazać wykresy z próbkami i wykrytymi ramkami
wrt : bool = False
dbg : bool = True
del_files : bool = False


samples_dir = Path ( "np.tensors" )

# Znajdź unikalne grupy plików na podstawie timestampu w nazwie dla plików *_rx_samples_*.npy które nie były jeszcze obrobione!!!
timestamp_groups = sorted ( { p.name.split ( "_rx_samples_" , 1 )[ 0 ] for p in Path("np.tensors").glob("*_rx_samples_*.npy") } )

for timestamp_group in timestamp_groups :
	samples_files = sorted ( samples_dir.glob ( f"{timestamp_group}_rx_samples_*.npy" ) )
	timestamps = sorted ( { p.stem.split(f"{timestamp_group}_rx_samples_", 1)[1] for p in Path("np.tensors").glob(f"{timestamp_group}_rx_samples_*.npy") } )
	if dbg : print ( f"\n{timestamp_group=} , {samples_files} , {timestamps=}" )
	if not samples_files :
		raise FileNotFoundError ( f"Brak plikow {timestamp_group}_rx_samples_*.npy w katalogu {samples_dir}" )
	rx_samples = packet.RxSamples_v0_1_18 ()
	for samples_file in samples_files :
		rx_samples.rx ( samples_filename = str ( samples_file ) , concatenate = True )
	if plt : rx_samples.plot_complex_samples ( f"{script_filename} raw samples {rx_samples.samples.size=}" )
	rx_samples.detect_frames ( deep = False , filter = True , correct = False , add_peak_at_0 = False )
	rx_samples_frame_start_idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_idx for frame in rx_samples.frames ] , dtype = np.uint32 )
	if plt : rx_samples.plot_complex_samples_corrected ( title = f"before cliping {script_filename} {rx_samples.samples.size=} {rx_samples_frame_start_idx.size=}" , peaks = rx_samples_frame_start_idx )
	if clp :
		rx_samples.clip_samples_for_training ()
		#rx_samples_frame_start_idx = np.array ( [ frame.frame_start_abs_idx for frame in rx_samples.frames ] , dtype = np.uint32 )
		if plt : rx_samples.plot_complex_samples ( f"{script_filename} {rx_samples.samples.size=}" , peaks = rx_samples_frame_start_idx )
		if plt : rx_samples.plot_complex_samples_corrected ( title = f"{script_filename} {rx_samples.samples.size=} {rx_samples_frame_start_idx.size=}" , peaks = rx_samples_frame_start_idx )
	print ( f"{rx_samples=}" )
	rx_symbols_flat_tensor = rx_samples.symbols_2_flat_tensor ()
	tx_symbols_flat_tensor : torch.Tensor = ops_file.open_flat_tensor ( file_name = f"{timestamp_group}_tx_symbols_flat_tensor.pt" , dir_name = samples_dir.name )
	tx_samples_flat_tensor : torch.Tensor = ops_file.open_flat_tensor ( file_name = f"{timestamp_group}_tx_samples_flat_tensor.pt" , dir_name = samples_dir.name )
	# Wariant 1. Sprawdzenie pełnej zgodności tensorów rx i tx. Jeśli są identyczne, to można bezpiecznie zapisać y_train i usunąć pliki z próbkami, bo nie będą już potrzebne do treningu.
	if torch.equal ( rx_symbols_flat_tensor , tx_symbols_flat_tensor ) :
		if wrt :
			rx_samples.save_frames2y_train_tensor ( file_name = f"{timestamp_group}_y_train_tensor" , dir_name = samples_dir.name )
			rx_samples.save_complex_samples2npf_v0_1_18 ( file_name = f"{timestamp_group}_rx_samples" , dir_name = samples_dir.name , add_timestamp = False )
			if del_files :
				for file_path in Path ( samples_dir ).glob ( f"{timestamp_group}_rx_samples_*.npy" ) :
					if file_path.is_file () :
						file_path.unlink ( missing_ok = True )
	else :
		if rx_samples.frames is not None :
			tx_samples = packet.RxSamples_v0_1_18 ()
			tx_samples.rx ( samples_filename = str ( f"{samples_dir.name}/{timestamp_group}_tx_samples4pluto.npy" ) , concatenate = True )
			tx_samples.detect_frames ( deep = False , filter = True , correct = False , add_peak_at_0 = True )
			tx_frames_start_idx = np.array ( [ frame.frame_start_abs_idx for frame in tx_samples.frames ] , dtype = np.uint32 )
			if plt : tx_samples.plot_complex_samples_corrected_v0_1_20 ( title = f"{script_filename} {rx_samples.samples.size=} {rx_samples_frame_start_idx.size=}" )
			if plt : plot.flat_tensor_v0_1_18 ( tx_symbols_flat_tensor , title = f"{script_filename} {timestamp_group} tx symbols flat tensor" , marker_idx = tx_frames_start_idx )
			if plt : plot.flat_tensor_v0_1_18 ( tx_samples_flat_tensor , title = f"{script_filename} {timestamp_group} tx samples flat tensor" , marker_idx = tx_frames_start_idx )

			'''
			W rx_samples znajdź frame, która jest identyczna jak pierwsza frame w tx_samples (czyli pierwsza ramka w pliku {timestamp_group}_tx_samples4pluto.npy)
			i zapisz jej pozycję początkową frame_starts_idx jako tx_rx_frame_start_abs_idx.
			'''
			rx_frames_start_abs_idx = None
			if tx_samples.frames is not None and len ( tx_samples.frames ) > 0 :
				for tx_unique_frame in tx_samples.frames :
					#if dbg : print ( f"\n\r{tx_unique_frame.frame_start_abs_idx=}" )
					tx_first_unique_frame = None
					is_unique = True
					for tx_frame in tx_samples.frames :
						#if dbg : print ( f",{tx_frame.frame_start_abs_idx=}" )
						if np.array_equal ( tx_unique_frame.header_bits , tx_frame.header_bits ) and ( tx_unique_frame.frame_start_abs_idx != tx_frame.frame_start_abs_idx ) :
							is_unique = False
							#if dbg : print ( f"{is_unique=}" )
					if is_unique :
						tx_first_unique_frame = tx_unique_frame
						for rx_frame in rx_samples.frames :
							if np.array_equal ( rx_frame.header_bits , tx_first_unique_frame.header_bits ) :
								#rx_frames_start_abs_idx = rx_frame.frame_start_abs_idx - ( tx_first_unique_frame.frame_start_abs_idx - filters.SPAN * modulation.SPS // 2 )
								rx_frames_start_abs_idx = rx_frame.frame_start_abs_idx - tx_first_unique_frame.frame_start_abs_idx
								break
						if rx_frames_start_abs_idx is not None :
							break
			if rx_frames_start_abs_idx is not None :
				print ( f"Znaleziono dopasowanie ramki: {timestamp_group} rx_frames_start_abs_idx={rx_frames_start_abs_idx}" )
				'''Znalazłem dopasowanie ramki w jednym z punktów SPS, ale nie wiem czy pierwszy. A chciałbym precyzyjnie znaleźć pierwszy, żeby mieć pewność,
				że plik y_train będzie idealnie dopasowany względem X_train. I nie będzie przesunięty ani o 1 sample.
				Będę szukał dopasowania frame header w 4 punktach, które są oddalone o 1 sample od siebie, czyli
				znaleźć precyzyjnie pierwszy sample a nie któryś z 4, który rozpoczyna frame
				Zacznę od cofania o 1 sample od znalezionego dopasowania, żeby znaleć ten pierwszy sample, a póżniej będę szedł do przodu o 1 sample,
				żeby znaleźć ostatni i czy wychodzi liczba sampli z dostposowaniem= SPS,
				'''



				'''Stworzenie symboli bpsk reprezentujacych cały przebieg rx_samples.samples:
				1. Na początku tworzymy rx_tensor i na całej długości wszędzie wstawiamy 0+j0
				2. Od pozycji rx_frames_start_abs_idx w rx_tensor podstawiamy odpowiednie symbole BPSK z tx_samples_flat_tensor.
				'''
				rx_tensor = torch.zeros ( rx_samples.samples.size , dtype = torch.complex64 )
				rx_tensor [ rx_frames_start_abs_idx : rx_frames_start_abs_idx + tx_samples_flat_tensor.size(0) ] = tx_samples_flat_tensor
				if plt : plot.flat_tensor_v0_1_18 ( rx_tensor , title = f"{script_filename} {timestamp_group} rx tensor aligned to tx symbols" , marker_idx = rx_samples_frame_start_idx )
				# Teraz można bezpiecznie zapisać y_train, bo jest pewność, że jest poprawnie dopasowane do X_train, a także można usunąć pliki z próbkami, bo nie będą już potrzebne do treningu.
				if wrt :
					rx_samples.save_frames2y_train_tensor ( file_name = f"{timestamp_group}_y_train_tensor" , dir_name = samples_dir.name )
					rx_samples.save_complex_samples2npf_v0_1_18 ( file_name = f"{timestamp_group}_rx_samples" , dir_name = samples_dir.name , add_timestamp = False )
					if del_files :
						for file_path in Path ( samples_dir ).glob ( f"{timestamp_group}_rx_samples_*.npy" ) :
							if file_path.is_file () :
								file_path.unlink ( missing_ok = True )
			else :
				print ( f"Nie znaleziono dopasowania ramki: {timestamp_group}" )
			pass