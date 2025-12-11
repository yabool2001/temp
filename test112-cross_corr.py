from modules import ops_file , modulation , packet , plot
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path

Path ( "logs" ).mkdir ( parents = True , exist_ok = True )
script_filename = os.path.basename ( __file__ )

filename_samples = "np.samples/complex_samples_1k.npy"
filename_sync_sequence = "np.samples/complex_sync_sqeuence_4_1k.npy"

samples  = ops_file.open_samples_from_npf ( filename_samples )
sync_sequence = ops_file.open_samples_from_npf ( filename_sync_sequence )

plot.complex_waveform ( samples , f"{script_filename} | {samples.size=}" , False )
plot.complex_waveform ( sync_sequence , f"{script_filename} | {sync_sequence.size=}" , False )
m = len ( samples )
n = len ( sync_sequence )
corr = np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ; plot.complex_waveform ( corr , f"{script_filename} | {corr.size=}" , False )
corr_abs = np.abs ( np.correlate ( samples , np.flip ( sync_sequence.conj () ) , mode = 'valid' ) ) ; plot.real_waveform ( corr_abs , f"{script_filename} | {corr_abs.size=}" , False )

sync_sequence_power = np.abs ( sync_sequence ) ** 2 ; plot.real_waveform ( sync_sequence_power , f"{script_filename} | {sync_sequence_power.size=}" , False )
sync_sequence_energy = np.sum ( sync_sequence_power ) ; print ( f" {sync_sequence_energy.size=} {sync_sequence_energy=}" )