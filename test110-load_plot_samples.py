from modules import ops_file , modulation , packet
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename9 = "logs/rx_samples_32768_9_empty.npy"
filename8 = "logs/rx_samples_32768_8_11samples.npy"
filename4 = "logs/rx_samples_32768_4.npy"
filename5 = "logs/rx_samples_32768_5.npy"

samples9 : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename9 )
#samples8 : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename8 )
#samples4 : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename4 )
#samples5 : NDArray[np.complex128] = ops_file.open_samples_from_npf ( filename5 )
#samples6: NDArray[np.complex128] = np.concatenate ( ( samples4 , samples5 ) )
rx_packets9 = packet.RxPackets ( samples = samples9 )
#rx_packets8 = packet.RxPackets ( samples = samples8 )
#rx_packets4 = packet.RxPackets ( samples = samples4 )
#rx_packets5 = packet.RxPackets ( samples = samples5 )
#rx_packets6 = packet.RxPackets ( samples = samples6 )
#rx_packets7.plot_waveform ( samples7 , "RX Samples 7" )
rx_packets9.plot_waveform ( samples9 , "RX Samples 9" )
#rx_packets4.plot_waveform ( rx_packets4.samples , "RX Samples4" )
#rx_packets5.plot_waveform ( rx_packets5.samples , "RX Samples5" )
#rx_packets6.plot_waveform ( rx_packets6.samples , "RX Samples6" )