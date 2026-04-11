# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file, packet , plot

script_filename = os.path.basename ( __file__ )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

s1 : list = []
s1.append ( "np.samples/rx_samples_0.1.18_4Bx2.npy" )

#samples_dir = Path ( "np.simple-frames" )
samples_dir = Path ( "np.samples_series_01" )
samples_files = sorted ( samples_dir.glob ( "*.npy" ) )
if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {samples_dir}" )

rx_pluto_samples = packet.RxSamples_v0_1_18 ()
for samples_file in samples_files :
	rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = True )
rx_pluto_samples.plot_complex_samples ( f"{script_filename} raw samples {rx_pluto_samples.samples.size=}" )
rx_pluto_samples.detect_frames ( deep = False , filter = True , correct = True )
rx_pluto_samples.plot_complex_samples_corrected ( title = f"{script_filename} concatenated samples {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
print ( f"{rx_pluto_samples.frames=}" )
for frame in rx_pluto_samples.frames :
	print ( f"{ frame.packet.payload_bytes=}, {frame.packet.payload_bytes.size=}" )
