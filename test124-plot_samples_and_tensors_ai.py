from pathlib import Path

import numpy as np , os
import torch
from numpy.typing import NDArray

from modules import packet , plot , ops_file

script_filename = os.path.basename ( __file__ )

dbg : bool = True
plt : bool = True
obj : bool = True # obj=object czy chcesz to robic za pomoca klas w modules/packet.py czy tylko funkcji w modules/ops_file.py

X_train_samples_file_name = "pt.inference/1781194806741_X_train_samples.npy"
y_train_tensor_file_name = "pt.inference/1781194806741_y_train_tensor.pt"
ai_samples_file_name = "np.demod/1781194806741_ai_demod_samples.npy"
ai_symbols_file_name = "np.demod/1781194806741_ai_symbols.npy"
timestamp_group = X_train_samples_file_name.split ( "_X_train_samples" , 1 )[ 0 ]
first_symbol_abs_idx = 66196
#last_symbol_abs_idx = 40470
last_symbol_abs_idx = first_symbol_abs_idx + 1000

if obj :
	X_train_samples = packet.RxSamples ()
	X_train_samples.rx ( file_name = str ( X_train_samples_file_name ) , concatenate = False )
	X_train_samples.detect_frames ( deep = False , samples_filtered = True , correct_samples = False )
	idxs = X_train_samples.aggregate_frame_and_packet_idxs ()
else :
	samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( X_train_samples_file_name ) )
	if dbg : print ( f"\n{X_train_samples_file_name} samples.shape = {samples.shape} samples.dtype = {samples.dtype}" )
ai_samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( ai_samples_file_name ) )
ai_symbols : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( ai_symbols_file_name ) )
y_train_tensor = torch.load ( y_train_tensor_file_name )

plot.samples_and_tensor_1k ( X_train_samples = X_train_samples.samples_raw[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							y_train_tensor = y_train_tensor[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							ai_samples = ai_samples[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							ai_symbols = ai_symbols[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							idxs = idxs - first_symbol_abs_idx ,
							my_title = f"{script_filename} {timestamp_group} {first_symbol_abs_idx=}-{last_symbol_abs_idx=}" )
plot.samples_and_tensor_1k ( X_train_samples = X_train_samples.samples_raw[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							ai_samples = ai_samples[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							idxs = idxs - first_symbol_abs_idx ,
							my_title = f"{script_filename} {timestamp_group} {first_symbol_abs_idx=}-{last_symbol_abs_idx=}" )
plot.samples_and_tensor_1k ( y_train_tensor = y_train_tensor[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							ai_symbols = ai_symbols[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							idxs = idxs - first_symbol_abs_idx ,
							my_title = f"{script_filename} {timestamp_group} {first_symbol_abs_idx=}-{last_symbol_abs_idx=}" )
plot.samples_and_tensor_1k ( X_train_samples = X_train_samples.samples_raw[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							ai_samples = ai_samples[ first_symbol_abs_idx : last_symbol_abs_idx ] ,
							idxs = idxs - first_symbol_abs_idx ,
							my_title = f"{script_filename} {timestamp_group} {first_symbol_abs_idx=}-{last_symbol_abs_idx=}" )
