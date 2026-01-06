from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename = "logs/rx_samples_32768_10.npy"
filename2 = "np.samples/rx_samples_log.npy"
samples : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename2 )
plot.complex_waveform_v0_1_6 ( samples , f"{samples.size=}" , marker_squares = False )
