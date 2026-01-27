from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray

samples_filename = "np.samples/rx_samples_0.1.14_1500B_01.npy"

samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( samples_filename )
plot.complex_waveform_v0_1_6 ( samples , f"{samples_filename} {samples.size=}" )
