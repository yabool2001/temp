from modules import ops_file , modulation , packet , plot
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename_samples = "np.samples/complex_samples_1k.npy"
filename_sync_sequence = "np.samples/complex_sync_sqeuence_4_1k.npy"

samples  = ops_file.open_samples_from_npf ( filename_samples )
rx_packets = packet.RxPackets ( samples = samples )
rx_packets.clip_samples ( 47 , 96 )
rx_packets.plot_waveform ( rx_packets.samples , "Samples clipped" )
#ops_file.save_samples_2_npf ( filename , rx_packets.samples )


ops_file.save_samples_2_npf(filename_sync_sequence, rx_packets.samples )
