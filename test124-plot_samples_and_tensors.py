from pathlib import Path

import numpy as np , os
import torch
from numpy.typing import NDArray

from modules import packet , plot , ops_file

script_filename = os.path.basename ( __file__ )

dbg : bool = True
plt : bool = True
obj : bool = True # obj=object czy chcesz to robic za pomoca klas w modules/packet.py czy tylko funkcji w modules/ops_file.py

dir_name = "training"
samples_file_name = "1778613855045_tx_samples4pluto.npy"
tensor_file_name = "1778613855045_y_train.pt"

samples_file = f"{dir_name}/{samples_file_name}"
tensor_file = f"{dir_name}/{tensor_file_name}"

if obj :
	rx_pluto_samples = packet.RxSamples_v0_1_18 ()
	rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = False )
	rx_pluto_samples.detect_frames ( deep = False , filter = True , correct = True )
else :
	samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file ) )
	if dbg : print ( f"\n{samples_file.name} samples.shape = {samples.shape} samples.dtype = {samples.dtype}" )

loaded_tensor = torch.load ( tensor_file )

plot.samples_and_tensor ( rx_pluto_samples.samples , loaded_tensor , tensor_m = 500 , title = "combined" , parts = 8 )
