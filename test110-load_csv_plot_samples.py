from modules import ops_file , packet
import numpy as np
import os
from numpy.typing import NDArray
from pathlib import Path

script_filename = os.path.basename ( __file__ )
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename1 = "samples.csv/rx_samples_log_1768037934826.csv"
filename2 = "samples.csv/rx_samples_log_1768037934826_clipped.csv"
filename_barker13_clipped_74 = "logs/tx_samples_barker13_clipped_74.csv"

wrt = False

rx_pluto = packet.RxPluto_v0_1_10 ()
rx_pluto.samples.rx ( samples_filename = filename1 )
rx_pluto.samples.plot_complex_samples ( title = f"{script_filename}" )
rx_pluto.samples.clip_samples ( start = 0 , end = 3000 )
if wrt :
    rx_pluto.samples.save_complex_samples_2_csv ( filename = filename2 )
    rx_pluto.samples.plot_complex_samples ( title = f"{script_filename}" )
if rx_pluto.samples.has_amp_greater_than_ths :
    rx_pluto.samples.detect_frames ()
