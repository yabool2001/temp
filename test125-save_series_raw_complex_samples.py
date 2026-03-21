from modules import packet , sdr
from pathlib import Path
import numpy as np
import os , sys , tomllib
import tomllib

Path ( "np.samples_series_01" ).mkdir ( parents = True , exist_ok = True )

plt = True
wrt = True

filename = "np.samples_series_01/rx_samples.npy"
series_len = 10

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

if len ( sys.argv ) > 1 :
    n_o_bytes_uint16 = np.uint16 ( int ( sys.argv[ 1 ] ) )
    n_o_repeats_uint32 = np.uint32 ( int ( sys.argv[ 2 ] ) )
    tx_gain_float = float ( sys.argv[ 3 ] )
else :
    n_o_bytes_uint16 = np.uint16 ( 4 )
    n_o_repeats_uint32 = np.uint32 ( 10 )
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

rx_pluto = packet.RxPluto_v0_1_16 ( sn = sdr.PLUTO_RX_SN )
# print ( f"\n{ script_filename= } receiving: {rx_pluto=} { rx_pluto.samples.samples.size= }" )

while series_len > 0 :
    series_len -= 1
    rx_pluto.samples.rx ()
    if wrt :
        rx_pluto.samples.save_complex_samples_2_npf ( filename )
        