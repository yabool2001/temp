# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

import numpy as np , os , tomllib , torch
from pathlib import Path

from numpy.typing import NDArray

from modules import ops_file, packet

script_filename = os.path.basename ( __file__ )
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

clp : bool = True # Czy przyciąć próbki do długości ramki (wymagane do treningu, ale nie do analizy)
plt : bool = False
wrt : bool = True
dbg : bool = True
del_files : bool = False


samples_dir = Path ( "np.tensors" )

# Znajdź unikalne grupy plików na podstawie timestampu w nazwie dla plików *_rx_samples_*.npy które nie były jeszcze obrobione!!!
timestamp_groups = sorted ( { p.name.split ( "_rx_samples" , 1 )[ 0 ] for p in Path("np.tensors").glob("*_rx_samples_*.npy") } )

for timestamp_group in timestamp_groups :
	samples_files = sorted ( samples_dir.glob ( f"{timestamp_group}_rx_samples_*.npy" ) )
	timestamps = sorted ( { p.stem.split(f"{timestamp_group}_rx_samples_", 1)[1] for p in Path("np.tensors").glob(f"{timestamp_group}_rx_samples_*.npy") } )
	if dbg : print ( f"\n{timestamp_group=} , {samples_files=} , {timestamps=}" )
	if not samples_files :
		raise FileNotFoundError ( f"Brak plikow {timestamp_group}_rx_samples_*.npy w katalogu {samples_dir}" )
	rx_pluto_samples = packet.RxSamples_v0_1_18 ()
	for samples_file in samples_files :
		rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = True )
	if plt : rx_pluto_samples.plot_complex_samples ( f"{script_filename} raw samples {rx_pluto_samples.samples.size=}" )
	rx_pluto_samples.detect_frames ( deep = False , filter = True , correct = True )
	frame_starts_idx : NDArray [ np.uint32 ] = np.array ( [ frame.frame_start_abs_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 )
	if plt : rx_pluto_samples.plot_complex_samples_corrected ( title = f"before cliping {script_filename} {rx_pluto_samples.samples.size=} {frame_starts_idx.size=}" , peaks = frame_starts_idx )
	if clp :
		rx_pluto_samples.clip_samples_for_training ()
		frame_starts_idx = np.array ( [ frame.frame_start_abs_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 )
		if plt : rx_pluto_samples.plot_complex_samples ( f"{script_filename} {rx_pluto_samples.samples.size=}" , peaks = frame_starts_idx )
		if plt : rx_pluto_samples.plot_complex_samples_corrected ( title = f"{script_filename} {rx_pluto_samples.samples.size=} {frame_starts_idx.size=}" , peaks = frame_starts_idx )
	#print ( f"{rx_pluto_samples.frames=}" )
	#for frame in rx_pluto_samples.frames :
	#	frame_symbols = np.concatenate ( [ frame.header_bpsk_symbols , frame.packet.bpsk_symbols ] )
	#	print ( f"{ frame_symbols.size=}, {frame.frame_start_abs_idx=}, {frame_symbols[ : 5 ]=}" )
	flat_tensor_rx = rx_pluto_samples.symbols_2_flat_tensor ()
	flat_tensor_tx : torch.Tensor = ops_file.open_flat_tensor ( file_name = f"{timestamp_group}_tx_symbols_flat_tensor.pt" , dir_name = samples_dir.name )
	if torch.equal ( flat_tensor_rx , flat_tensor_tx ) :
		if wrt :
			rx_pluto_samples.save_frames2y_train_tensor ( file_name = f"{timestamp_group}_y_train_tensor" , dir_name = samples_dir.name )
			rx_pluto_samples.save_complex_samples2npf_v0_1_18 ( file_name = f"{timestamp_group}_rx_samples" , dir_name = samples_dir.name , add_timestamp = False )
			if del_files :
				for file_path in Path ( samples_dir ).glob ( f"{timestamp_group}_rx_samples_*.npy" ) :
					if file_path.is_file () :
						file_path.unlink ( missing_ok = True )