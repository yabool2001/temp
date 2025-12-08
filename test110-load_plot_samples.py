from modules import ops_file , modulation , packet
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename4 = "logs/rx_samples_32768_4.npy"
filename5 = "logs/rx_samples_32768_5.npy"

samples4 : NDArray[np.complex128] = ops_file.open_samples_from_npfile_v0_1_6 ( filename4 )
samples5 : NDArray[np.complex128] = ops_file.open_samples_from_npfile_v0_1_6 ( filename5 )
samples6: NDArray[np.complex128] = np.concatenate ( ( samples4 , samples5 ) )
rx_packets4 = packet.RxPackets ( samples = samples4 )
rx_packets5 = packet.RxPackets ( samples = samples5 )
rx_packets6 = packet.RxPackets ( samples = samples6 )
rx_packets4.plot_waveform ( rx_packets4.samples , "RX Samples4" )
rx_packets5.plot_waveform ( rx_packets5.samples , "RX Samples5" )
rx_packets6.plot_waveform ( rx_packets6.samples , "RX Samples6" )