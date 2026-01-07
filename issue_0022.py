
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.8-tx.py

Po zakończeniu tej wersji wrócić do rozwoju wersji bpsk_v0.1.6-rx
'''
import numpy as np
from numpy.typing import NDArray
import os
import time as t
import tomllib
from pathlib import Path
from numpy import real

#from pathlib import Path
from modules import packet , ops_file

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )
filename = "np.samples/rx_samples_no_peak_in_9th_frame.npy"
filename_csv = "np.samples/rx_samples_no_peak_in_9th_frame.csv"
rx_pluto = packet.RxPluto_v0_1_10 ()
rx_pluto.samples.rx ( samples_filename = filename )
#print(f"Max abs w surowych próbkach: {np.max(np.abs(rx_pluto.samples.samples))}")
rx_pluto.samples.plot_complex_samples ( title = f"{script_filename}" )
rx_pluto.samples.detect_frames ()

print ( f" {rx_pluto.samples.frames.samples_payloads_bytes=}, {rx_pluto.samples.frames.samples_payloads_bytes.size=}" )

