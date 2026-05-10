from pathlib import Path

import numpy as np , os
import torch
from numpy.typing import NDArray

from modules import plot

script_filename = os.path.basename ( __file__ )

dir_name = Path ( "np.tensors" )
dir_name = Path ( "training" )

tensor_files = sorted ( dir_name.glob ( "*.pt" ) )

if not tensor_files :
	raise FileNotFoundError ( f"Brak plikow .pt w katalogu {dir_name}" )

for tensor_file in tensor_files :
	loaded_tensor = torch.load ( tensor_file )
	if not isinstance ( loaded_tensor , torch.Tensor ) :
		raise TypeError ( f"Plik {tensor_file} nie zawiera torch.Tensor" )
	if "flat" in tensor_file.name.lower() :
		plot.flat_tensor_v0_1_18 ( loaded_tensor , f"{script_filename} | {tensor_file.name} flat tensor {loaded_tensor.size()}" )
	elif "y_train" in tensor_file.name.lower() :
		plot.y_train_tensor_as_flat_tensor_v0_1_18 ( y_train_tensor = loaded_tensor , title = f"{script_filename} | {tensor_file.name} tensor {loaded_tensor.size()}" )
