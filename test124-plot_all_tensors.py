from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from modules import plot

dir_name = Path ( "np.tensors" )

tensor_files = sorted ( dir_name.glob ( "*.pt" ) )

if not tensor_files :
	raise FileNotFoundError ( f"Brak plikow .pt w katalogu {dir_name}" )

for tensor_file in tensor_files :
	loaded_tensor = torch.load ( tensor_file )
	if not isinstance ( loaded_tensor , torch.Tensor ) :
		raise TypeError ( f"Plik {tensor_file} nie zawiera torch.Tensor" )
	if "flat" in tensor_file.name.lower() :
		plot.flat_tensor_v0_1_18 ( loaded_tensor , f"{tensor_file.name} flat tensor {loaded_tensor.size()}" )
	elif "y_train" in tensor_file.name.lower() :
		plot.y_train_tensor_as_flat_tensor_v0_1_18 ( loaded_tensor , f"{tensor_file.name} tensor {loaded_tensor.size()}" )
