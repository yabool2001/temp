# issue #45 - dekodowanie symboli z wszystkich plików npy zapisanych w danym katalogu
# i zapisywanie ich w y_train

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file, packet , plot

s1 = "np.samples_series_01/rx_samples_1774207593372.npy"

samples_dir = Path ( "np.samples_series_01" )
samples_files = sorted ( samples_dir.glob ( "*.npy" ) )
if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {samples_dir}" )

for samples_file in samples_files :
	rx_pluto_samples = packet.RxSamples_v0_1_17 ()
	rx_pluto_samples.rx ( samples_filename = str ( samples_file ) )
	#rx_pluto_samples.detect_frames ( deep = False )