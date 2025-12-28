from modules import corrections , filters , modulation , ops_file , ops_packet , packet , plot , sdr
from pathlib import Path
import numpy as np
import os
import tomllib
from numpy.typing import NDArray

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

plt = True
wrt = True

filename = "np.samples/rx_samples_0.1.8_.npy"

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

pluto_rx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_RX"] )
for i in range ( 0 , 1 ) : # Clear buffer just to be safe
    raw_data = sdr.rx_samples ( pluto_rx )
print ( "Start Pluto Rx!" )
while True :
    raw_data = sdr.rx_samples ( pluto_rx )
    rx_samples = packet.RxSamples_v0_1_8 ( raw_data )
    if rx_samples.has_amp_greater_than_ths :
        print ( "Preambuła znaleziona!" )
        break
if plt :
    plot.complex_waveform_v0_1_6 ( raw_data , f"{raw_data.size=}" , False )
if wrt :
    ops_file.save_complex_samples_2_npf ( filename , raw_data )
        