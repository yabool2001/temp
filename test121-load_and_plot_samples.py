from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray

samples_filename = "np.samples/rx_samples_0.1.14_1500B_01.npy"
samples_filename1 = "np.samples/rx_samples_0.1.17_issue40_001.0.npy"
samples_filename2 = "np.samples/rx_samples_0.1.17_issue40_001.1.npy"
samples_filename3 = "np.samples/rx_samples_0.1.17_1770460766353.npy"


samples3 : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( samples_filename3 )
plot.complex_waveform_v0_1_6 ( samples3 , f"{samples_filename3} {samples3.size=}" )
