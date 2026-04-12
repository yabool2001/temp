# issue #45 - dekodowanie symboli z wszystkich plików X-train npy zapisanych we wskazanym katalogu
# i zapisywanie ich jako y_train w plikach odpowiadajacych X_train

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from modules import ops_file, packet , plot

script_filename = os.path.basename ( __file__ )

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

#samples_dir = Path ( "np.simple-frames" )
#samples_dir = Path ( "np.samples_test127" )
samples_dir = Path ( "np.tensor_1x1500B" )
samples_files = sorted ( samples_dir.glob ( "*.npy" ) )
#dir_name = "np.tensors_001"
#dir_name = "np.samples_test127"

if not samples_files :
	raise FileNotFoundError ( f"Brak plikow .npy w katalogu {samples_dir}" )

#samples_files : list = [ "np.samples/rx_samples_1776002831362.npy" ] # do testowania na jednym pliku, żeby szybciej iterować nad poprawkami
#samples_files : list = [ "np.tensors_001/rx_samples_1776012929739.npy" ]

rx_pluto_samples = packet.RxSamples_v0_1_18 ()
for samples_file in samples_files :
	rx_pluto_samples.rx ( samples_filename = str ( samples_file ) , concatenate = True )
rx_pluto_samples.plot_complex_samples ( f"{script_filename} raw samples {rx_pluto_samples.samples.size=}" )
rx_pluto_samples.detect_frames ( deep = False , filter = True , correct = True )
#rx_pluto_samples.plot_complex_samples ( title = f"{script_filename} {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
#rx_pluto_samples.plot_complex_samples_filtered ( title = f"{script_filename} {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
rx_pluto_samples.plot_complex_samples_corrected ( title = f"{script_filename} {rx_pluto_samples.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.sync_sequence_peaks )
rx_pluto_samples.plot_y_train_tensor ( title = f"{script_filename} y_train_tensor {rx_pluto_samples.y_train_tensor.size=}" )
print ( f"{rx_pluto_samples.frames=}" )
for frame in rx_pluto_samples.frames :
	frame_symbols = np.concatenate ( [ frame.header_bpsk_symbols , frame.packet.bpsk_symbols ] )
	print ( f"{ frame_symbols.size=}, {frame.frame_start_abs_idx=}, {frame_symbols[ : 5 ]=}" )
flat_tensor_rx = rx_pluto_samples.flat_tensor_from_frames ( )
flat_tensor_tx : torch.Tensor = ops_file.open_flat_tensor ( file_name = "1776029890813.pt" , dir_name = samples_dir.name )
#flat_tensor_tx : torch.Tensor = ops_file.open_flat_tensor ( file_name = "1776012846025.pt" , dir_name = dir_name )
print ( f"{torch.equal ( flat_tensor_rx , flat_tensor_tx )=}" )