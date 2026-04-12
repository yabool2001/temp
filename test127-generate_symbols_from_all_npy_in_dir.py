# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modules import ops_file, packet , plot

script_filename = os.path.basename ( __file__ )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

#samples_dir = Path ( "np.simple-frames" )
samples_dir = Path ( "np.samples_test127" )
samples_files = sorted ( samples_dir.glob ( "*.npy" ) )
if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {samples_dir}" )

samples_files : list = [ "np.samples/rx_samples_1776002831362.npy" ] # do testowania na jednym pliku, żeby szybciej iterować nad poprawkami
samples_files : list = [ "np.samples_test127/1775927999910_rx_samples_1775928030100.npy" ]

rx_pluto_samples = packet.RxSamples_v0_1_18 ()
for samples_file in samples_files :
	rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = True )
rx_pluto_samples.plot_complex_samples ( f"{script_filename} raw samples {rx_pluto_samples.samples.size=}" )
rx_pluto_samples.detect_frames ( deep = False , filter = True , correct = True )
#rx_pluto_samples.detect_frames ( deep = False , filter = False , correct = False )
rx_pluto_samples.plot_complex_samples ( title = f"{script_filename} {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
rx_pluto_samples.plot_complex_samples_filtered ( title = f"{script_filename} {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
rx_pluto_samples.plot_complex_samples_corrected ( title = f"{script_filename} {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
print ( f"{rx_pluto_samples.frames=}" )
for frame in rx_pluto_samples.frames :
	frame_symbols = np.concatenate ( [ frame.header_bpsk_symbols , frame.packet.bpsk_symbols ] )
	print ( f"{ frame_symbols.size=}, {frame.frame_start_abs_idx=}, {frame_symbols[ : 5 ]=}" )
