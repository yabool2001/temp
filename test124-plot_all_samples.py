import numpy as np , os
from numpy.typing import NDArray
from pathlib import Path
from modules import ops_file , packet , plot

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

dbg : bool = True
plt : bool = True
obj : bool = True # obj=object czy chcesz to robic za pomoca klas w modules/packet.py czy tylko funkcji w modules/ops_file.py

dir_name = Path ( "np.tensors" )
dir_name = Path ( "training" )
#dir_name = Path ( "np.tensors_002_inference" )
#dir_name = Path ( "np.tensors_003_inference" )

samples_files = sorted ( dir_name.glob ( "*.npy" ) )

if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {dir_name}" )

i = 0
for samples_file in samples_files :
	if obj :
		rx_pluto_samples = packet.RxSamples_v0_1_18 ()
		# filtrowanie i korekcja wyłączona, bo zastosowana 2 razy nie zadziała
		rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = False )
		rx_pluto_samples.detect_frames ( deep = False , filter = False , correct = False )
		#frame_starts_idx : NDArray[ np.uint32 ] = np.array ( [ frame.frame_start_abs_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 )
		#frame_first_sample_idx : NDArray[ np.uint32 ] = np.array ( [ frame.frame_first_sample_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 )
		#idxs = np.concatenate ( ( 
		#	np.array ( [ frame.frame_start_abs_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 ) ,
		#	np.array ( [ frame.frame_start_abs_first_sample_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 ) ) )
		if plt : rx_pluto_samples.plot_complex_samples ( title = f"{script_filename} {samples_file.name=}")
	else :
		samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file ) )
		if dbg : print ( f"\n{samples_file.name} samples.shape={samples.shape} samples.dtype={samples.dtype}" )
		if plt : plot.complex_waveform_v0_1_6 ( samples , f"{samples_file.name} samples.size={samples.size}")
	#i += 1
	if i >= 2 :
		break