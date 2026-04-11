from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file , plot


#dir_name = Path ( "np.samples" )
#dir_name = Path ( "np.samples_series_01" )
#dir_name = Path ( "np.simple-frames" )
dir_name = Path ( "np.tensors" )

tensor_files = sorted ( dir_name.glob ( "*.npy" ) )

if not tensor_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {dir_name}" )

for tensor_file in tensor_files :
	symbols : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( tensor_file ) )
	plot.plot_symbols ( symbols , f"{tensor_file.name} {symbols.size=}" )
	plot.complex_symbols_v0_1_6 ( symbols , f"{tensor_file.name}" )
