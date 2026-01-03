
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

from numpy import real

#from pathlib import Path
from modules import ops_file , packet

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

#filename = "np.samples/rx_samples_0.1.7_02_32768.npy"
#filename = "np.samples/rx_samples_0.1.8_08_1s_sat.npy"
#filename = "np.samples/rx_samples_0.1.8_11_1s_sat.npy"
#filename = "np.samples/rx_samples_0.1.8_13_1s_sat.npy"
#filename = "np.samples/rx_samples_0.1.8_15_c_mode.npy"
#filename = "np.samples/rx_samples_0.1.8_16_c_mode.npy"
filename = "np.samples/rx_samples_0.1.8_17_c_mode_full.npy"

real = False

if real :
    rx_pluto = packet.RxPluto_v0_1_9 ()
else :
    rx_pluto = packet.RxPluto_v0_1_9 ( samples_filename = filename )

print ( f"\n{ script_filename= } {rx_pluto=} { rx_pluto.samples.samples.size= }" )
rx_pluto.samples.rx ()
rx_pluto.samples.detect_frames ()
print ( f"\n{ script_filename= } { rx_pluto.samples.samples.size= } { rx_pluto.samples.samples_filtered.size= }" )
if rx_pluto.samples.frames.has_leftovers :
    print ( f"{ rx_pluto.samples.frames.samples_leftovers_start_idx= }" )
print ( f" {rx_pluto.samples.frames.samples_payloads_bytes=}" )