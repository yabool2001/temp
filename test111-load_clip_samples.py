from modules import ops_file , modulation , packet
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename = "logs/rx_samples_32768_10.npy"
filename9 = "logs/rx_samples_32768_3_1sample.npy"
samples : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename9 )
rx_packets = packet.RxPackets ( samples = samples )
rx_packets.plot_waveform ( rx_packets.samples , "RX Samples original" )
rx_packets.clip_samples ( 9900 , 10800 )
rx_packets.plot_waveform ( rx_packets.samples , "RX Samples clipped" )
#ops_file.save_samples_2_npf ( filename , rx_packets.samples )
