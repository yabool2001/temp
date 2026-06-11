from pathlib import Path

import numpy as np , os
import torch
from numpy.typing import NDArray

from modules import packet , plot , ops_file

script_filename = os.path.basename ( __file__ )

dbg : bool = True
plt : bool = True
obj : bool = True # obj=object czy chcesz to robic za pomoca klas w modules/packet.py czy tylko funkcji w modules/ops_file.py

samples_file_name = "pt.inference/1781011870731_X_train_samples.npy"
tensor_file_name = "pt.inference/1781011870731_y_train_tensor.pt"
first_symbol_abs_idx = 40139 + 20 # 20 to jest offset na zimny rozbieg ML/AI, który jest potrzebny, bo LSTM potrzebuje rozbiegu, żeby zacząć dobrze działać. Bez tego rozbiegu, pierwsze 2048 próbek (warmup) są "zjedzone" i nie mają wpływu na wynik, co powoduje, że pierwsze 20 symboli (20*4096/8192=10) jest zniekształconych. Ten offset 20 jest dobrany eksperymentalnie, żeby idealnie wyrównać początek ramek z idealnymi symbolami BPSK.
last_symbol_abs_idx = 40470

if obj :
	rx_pluto_samples = packet.RxSamples ()
	rx_pluto_samples.rx ( file_name = str ( samples_file_name ) , concatenate = False )
	rx_pluto_samples.detect_frames ( deep = False , samples_filtered = True , correct_samples = False )
else :
	samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file_name ) )
	if dbg : print ( f"\n{samples_file_name} samples.shape = {samples.shape} samples.dtype = {samples.dtype}" )

loaded_tensor = torch.load ( tensor_file_name )

plot.samples_and_tensor ( rx_pluto_samples.samples_raw[ first_symbol_abs_idx : last_symbol_abs_idx ] , loaded_tensor[ first_symbol_abs_idx : last_symbol_abs_idx ] , tensor_m = 5000 , title = "combined" , parts = 8 )
