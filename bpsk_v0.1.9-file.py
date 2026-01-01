
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.8-tx.py

Po zakończeniu tej wersji wrócić do rozwoju wersji bpsk_v0.1.6-rx
'''

import os
import time as t
import tomllib

#from pathlib import Path
from modules import ops_file , packet

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

#filename_samples = "np.samples/rx_samples_0.1.8_08_1s_sat.npy"
filename_samples = "np.samples/rx_samples_0.1.8_01_32768.npy"

rx_pluto = packet.RxPluto_v0_1_9 ( samples_filename = filename_samples )
print ( f"\n{ script_filename= } {rx_pluto=} { rx_pluto.samples= }" )
print ( f"\n{ script_filename= } { rx_pluto.samples= } { rx_pluto.samples.samples_filtered.size= } , dtype = { rx_pluto.samples.samples_filtered.dtype= }" )
if rx_pluto.samples.frames.has_leftovers :
    print ( f"{ rx_pluto.samples.frames.samples_leftovers_start_idx= }" )
print ( f" {rx_pluto.samples.frames.samples_payloads_bytes=}" )