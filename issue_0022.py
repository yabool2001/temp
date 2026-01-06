
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
#print(f"{np.max(np.abs(rx_pluto.samples.samples))}=")
rx_pluto.samples.detect_frames ()
#print(f"{np.max(np.abs(rx_pluto.samples.samples_filtered))}=")
#ops_file.write_samples_2_csv ( filename_csv , rx_pluto.samples.samples_filtered )
print(f"Rozmiar samples_filtered: {rx_pluto.samples.samples_filtered.size}")
if 17732 < rx_pluto.samples.samples_filtered.size:
    print(f"Wartość w idx 17732: {rx_pluto.samples.samples_filtered[17732]}")
    print(f"Abs wartość: {np.abs(rx_pluto.samples.samples_filtered[17732])}")
else:
    print("Idx 17732 poza zakresem!")

# Sprawdź, ile próbek ma abs > 7000
abs_values = np.abs(rx_pluto.samples.samples_filtered)
count_above_7000 = np.sum(abs_values > 7000)
print(f"Liczba próbek z abs > 7000: {count_above_7000}")
if count_above_7000 > 0:
    max_above_7000 = np.max(abs_values[abs_values > 7000])
    print(f"Największa abs > 7000: {max_above_7000}")
    indices_above_7000 = np.where(abs_values > 7000)[0]
    print(f"Indeksy próbek z abs > 7000: {indices_above_7000}")

rx_pluto.samples.plot_complex_samples_filtered ( title = f"{script_filename}" , peaks = np.array([17732]) )
print(f"Sync sequence peaks: {rx_pluto.samples.frames.sync_sequence_peaks}")
print(f"Czy 17732 jest w peaks? {17732 in rx_pluto.samples.frames.sync_sequence_peaks}")
print ( f" {rx_pluto.samples.frames.samples_payloads_bytes=}, {rx_pluto.samples.frames.samples_payloads_bytes.size=}" )

