from modules import ops_file , plot
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

plt= True
wrt = False

load_complex_samples_filename = "np.samples/rx_samples_0.1.14_128Bx20_missed_last_frames.npy"
save_samples_real_filename = "np.samples/rx_samples_0.1.14_128Bx20_missed_last_frames_real.npy"
save_samples_imag_filename = "np.samples/rx_samples_0.1.14_128Bx20_missed_last_frames_imag.npy"


complex_samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( load_complex_samples_filename )

if plt:
    plot.complex_waveform ( complex_samples , f"Complex samples {load_complex_samples_filename} {complex_samples.size=}" )
    plot.real_waveform ( complex_samples.real , f"Real part of samples {save_samples_real_filename} {complex_samples.real.size=}" )
    plot.real_waveform ( complex_samples.imag , f"Imaginary part of samples {save_samples_imag_filename} {complex_samples.imag.size=}" )

if wrt :
    ops_file.save_float64_samples_2_npf ( save_samples_real_filename , complex_samples.real )
    ops_file.save_float64_samples_2_npf ( save_samples_imag_filename , complex_samples.imag )