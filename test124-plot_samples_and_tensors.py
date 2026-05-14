from pathlib import Path

import numpy as np , os
import torch
from numpy.typing import NDArray

from modules import packet , plot , ops_file

script_filename = os.path.basename ( __file__ )

dbg : bool = True
plt : bool = True
obj : bool = True # obj=object czy chcesz to robic za pomoca klas w modules/packet.py czy tylko funkcji w modules/ops_file.py

dir_name = Path ( "training" )

samples_files = sorted ( dir_name.glob ( "*.npy" ) )
if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {dir_name}" )

tensor_files = sorted ( dir_name.glob ( "*.pt" ) )
if not tensor_files :
	raise FileNotFoundError ( f"Brak plikow .pt w katalogu {dir_name}" )

for tensor_file in tensor_files :
	loaded_tensor = torch.load ( tensor_file )
	if not isinstance ( loaded_tensor , torch.Tensor ) :
		raise TypeError ( f"Plik {tensor_file} nie zawiera torch.Tensor" )
	#plot.y_train_tensor_as_flat_tensor_v0_1_18 ( y_train_tensor = loaded_tensor , title = f"{script_filename} | {tensor_file.name} tensor {loaded_tensor.size()}" )

for samples_file in samples_files :
	if obj :
		rx_pluto_samples = packet.RxSamples_v0_1_18 ()
		rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = False )
		rx_pluto_samples.detect_frames ( deep = False , filter = False , correct = False )
		#frame_starts_idx : NDArray[ np.uint32 ] = np.array ( [ frame.frame_start_abs_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 )
		#frame_first_sample_idx : NDArray[ np.uint32 ] = np.array ( [ frame.frame_first_sample_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 )
		#idxs = np.concatenate ( ( 
		#	np.array ( [ frame.frame_start_abs_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 ) ,
		#	np.array ( [ frame.frame_start_abs_first_sample_idx for frame in rx_pluto_samples.frames ] , dtype = np.uint32 ) ) )
		#if plt : rx_pluto_samples.plot_complex_samples ( title = f"{script_filename} {samples_file.name=}")
	else :
		samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file ) )
		if dbg : print ( f"\n{samples_file.name} samples.shape={samples.shape} samples.dtype={samples.dtype}" )
		#if plt : plot.complex_waveform_v0_1_6 ( samples , f"{samples_file.name} samples.size={samples.size}")

plot.samples_and_tensor ( rx_pluto_samples.samples , loaded_tensor , tensor_m = 500 , title = "combined" , parts = 8 )