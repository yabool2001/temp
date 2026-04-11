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
	symbols : NDArray[ np.complex128 ] = loaded_tensor.detach ().cpu ().numpy ().astype ( np.complex128 , copy = False )
	plot.plot_symbols ( symbols , f"{tensor_file.name} {symbols.size=}" )
	plot.complex_symbols_v0_1_6 ( symbols , f"{tensor_file.name}" )
