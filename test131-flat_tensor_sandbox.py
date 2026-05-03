'''
Skrypt do generowania prostej ramkii zapisywania jej do pliku npy i pt.
Dane będą odczytywane w celu weryfikacji poprawności i gotowosci do wykorzystania przy tworzeniu plików treningowych
'''

import socket, numpy as np , os , sys , time as t , tomllib
from modules import filters, modulation, ops_os, packet , payload_test_data as ptd , plot
from pathlib import Path

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

np.set_printoptions ( threshold = 10 , edgeitems = 3 , linewidth = np.inf ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

debug = True
plt = True
wrt = False
del_old = True

tx_samples = []
tx_samples_4pluto = np.array ( [] , dtype = np.complex128 )

dir_name = "np.tensors"

if del_old :
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

i = 1 # Liczba ramek
tx_samples = packet.TxSamples_v0_1_18 ()
while i :
    payload_bytes = ptd.PAYLOAD_4BYTES_DEC_15
    tx_samples.add_frame ( payload_bytes = payload_bytes )
    i -= 1
tx_samples.samples4pluto_2_flat_tensor ()

print ( f"{tx_samples.samples4pluto.size=}" )
print ( f"{tx_samples.frames=}" )

 # Środek ramki
frame_middle = filters.SPAN * modulation.SPS // 2
print (f"{frame_middle=}")

# Początek ramki
frame_beggining = ( filters.SPAN * modulation.SPS // 2 ) - ( modulation.SPS//2 )
print (f"{frame_beggining=}")

timestamp = ops_os.milis_timestamp ()

if plt :
    tx_samples.plot_complex_samples4pluto ( f"{script_filename} tx samples4pluto" , marker_peaks = True )
    plot.flat_tensor_v0_1_18 ( flat_tensor = tx_samples.frames[0].symbols_flat_tensor , title = "symbols flat tensors stworzony w RxFrames" )
    tx_samples.plot_flat_tensor ( f"{script_filename} tx flat tensor" , marker_idx = True )
    plot.flat_tensor_v0_1_18 ( flat_tensor = tx_samples.frames[0].samples_flat_tensor , title = "samples flat tensors stworzony w RxFrames" )