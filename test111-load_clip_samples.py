from modules import ops_file , modulation , packet
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename = "logs/rx_samples_32768_7.npy"
filename8 = "logs/rx_samples_32768_8.npy"
samples : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename )
rx_packets = packet.RxPackets ( samples = samples )
rx_packets.plot_waveform ( rx_packets.samples , "RX Samples original" )
rx_packets.clip_samples ( 0 , 17300 )
rx_packets.plot_waveform ( rx_packets.samples , "RX Samples clipped" )
#ops_file.save_samples_2_npf ( filename8 , rx_packets.samples )
