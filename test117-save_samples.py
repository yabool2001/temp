from modules import ops_file , packet , plot , sdr
from pathlib import Path
import numpy as np
import os
import tomllib
from numpy.typing import NDArray

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

plt = True
wrt = False

filename = "np.samples/rx_samples_0.1.15_.npy"

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

received_bytes : NDArray[ np.uint8 ] = np.array ( [] , dtype = np.uint8 )
rx_pluto = packet.RxPluto_v0_1_13 ( sn = sdr.PLUTO_RX_SN )
print ( f"\n{ script_filename= } receiving: {rx_pluto=} { rx_pluto.samples.samples.size= }" )

while len ( received_bytes ) == 0 :
    rx_pluto.samples.rx ()
    rx_pluto.samples.detect_frames ()
    if rx_pluto.samples.frames.samples_payloads_bytes.size > 0 :
        print ( f" { rx_pluto.samples.frames.samples_payloads_bytes.size= }" )
        np.concatenate ( [ received_bytes , rx_pluto.samples.frames.samples_payloads_bytes ] )

if plt :
    plot.complex_waveform_v0_1_6 ( rx_pluto.samples.samples , f"{rx_pluto.samples.samples.size=}" , False )
if wrt :
    ops_file.save_complex_samples_2_npf ( filename , rx_pluto.samples.samples )
        