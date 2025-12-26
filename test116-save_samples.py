from modules import corrections , filters , modulation , ops_file , ops_packet , packet , plot , sdr
from pathlib import Path
import numpy as np
import os
import tomllib
from numpy.typing import NDArray

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

plt = True
wrt = True

filename = "np.samples/rx_samples_1.npy"

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

pluto_rx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_RX"] )
for i in range ( 0 , 100 ) : # Clear buffer just to be safe
    raw_data = sdr.rx_samples ( pluto_rx )
print ( "Start Pluto Rx!" )
while True :
    rx_samples = sdr.rx_samples ( pluto_rx )
    rx_samples_filtered = filters.apply_rrc_rx_filter_v0_1_6 ( rx_samples )
    samples = packet.RxFrame_v0_1_8 ( samples_filtered = rx_samples_filtered )
    if samples.has_sync_sequence :
        print ( "Preambuła znaleziona!" )
        break
if plt :
    plot.complex_waveform_v0_1_6 ( rx_samples , f"{rx_samples.size=}" , False )
if wrt :
    ops_file.save_complex_samples_2_npf ( filename , rx_samples )
        