from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray

samples_filename = "np.samples/rx_samples_0.1.14_1500B_01.npy"
samples_filename1 = "np.samples/rx_samples_0.1.17_issue40_001.0.npy"
samples_filename2 = "np.samples/rx_samples_0.1.17_issue40_001.1.npy"


samples1 : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( samples_filename1 )
samples2 : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( samples_filename2 )
plot.complex_waveform_v0_1_6 ( samples1 , f"{samples_filename1} {samples1.size=}" )
plot.complex_waveform_v0_1_6 ( samples2 , f"{samples_filename2} {samples2.size=}" )
