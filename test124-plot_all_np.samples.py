from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file , plot


samples_dir = Path ( "np.samples" )
samples_files = sorted ( samples_dir.glob ( "*.npy" ) )

if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {samples_dir}" )

for samples_file in samples_files :
	samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( samples_file ) )
	plot.complex_waveform_v0_1_6 ( samples , f"{samples_file.name} samples.size={samples.size}" )
