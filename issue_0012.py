import numpy as np

from modules import ops_file , packet
from numpy.typing import NDArray
from pathlib import Path

Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

saved_rx_packets_samples_filename = "logs/rx_samples_32768.csv"
saved_rx_barker13_samples_filename = "logs/barker13_samples.npy"
barker13_samples = ops_file.open_samples_from_npf ( saved_rx_barker13_samples_filename )
rx_packets_samples = ops_file.open_csv_and_load_np_complex128 ( saved_rx_packets_samples_filename )
rx_samples = ops_file.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )
