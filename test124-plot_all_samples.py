from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file , plot


dir_name = Path ( "np.tensors" )

samples_files = sorted ( dir_name.glob ( "*.npy" ) )

if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {dir_name}" )

for samples_file in samples_files :
	samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file ) )
	print ( f"\n{samples_file.name} samples.shape={samples.shape} samples.dtype={samples.dtype}" )
	plot.complex_waveform_v0_1_6 ( samples , f"{samples_file.name} samples.size={samples.size}" )
