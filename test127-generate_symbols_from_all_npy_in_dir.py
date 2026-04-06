# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file, packet , plot

script_filename = os.path.basename ( __file__ )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

s1 = "np.samples_series_01/rx_samples_1774207593372.npy"

samples_dir = Path ( "np.samples_series_01" )
samples_files = sorted ( samples_dir.glob ( "*.npy" ) )
if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {samples_dir}" )

rx_pluto_samples = packet.RxSamples_v0_1_18 ()

# do usunięcia
rx_pluto_samples.rx ( samples_filename = "np.samples/rx_samples_0.1.14_128Bx20.npy" )
rx_pluto_samples.plot_complex_samples ( title = f"{ script_filename}" )
print ( f"{ rx_pluto_samples.samples.size=}" )
rx_pluto_samples.detect_frames ( deep = False )


for samples_file in samples_files :
	rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = True )
#plot.complex_waveform_v0_1_6 ( rx_pluto_samples.samples , f"concatenated samples {rx_pluto_samples.samples.size=}" )
rx_pluto_samples.detect_frames ( deep = False )
rx_pluto_samples.plot_complex_samples_filtered ( title = f"{script_filename} concatenated samples {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
print ( f"{rx_pluto_samples.frames=}" )
for frame in rx_pluto_samples.frames :
	print ( f"{ frame.payload_bytes=}, {frame.samples_payload_bytes.size=}" )
