from modules import ops_file , packet
import numpy as np
import os
from numpy.typing import NDArray
from pathlib import Path

script_filename = os.path.basename ( __file__ )
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename1 = "logs/rx_samples_32768.csv"
filename_barker13_clipped_74 = "logs/tx_samples_barker13_clipped_74.csv"

rx_pluto = packet.RxPluto_v0_1_10 ()
rx_pluto.samples.rx ( samples_filename = filename1 )
rx_pluto.samples.plot_complex_samples ( title = f"{script_filename}" )
rx_pluto.samples.clip_samples ( start = 0 , end = 2000 )
if rx_pluto.samples.has_amp_greater_than_ths :
    rx_pluto.samples.detect_frames ()
