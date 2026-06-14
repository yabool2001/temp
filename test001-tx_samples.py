# Skrypt do testowanie klasy TxSamples, TxFrame i TxSymbol, które służą do generowania próbek do transmisji.

import numpy as np , os , socket , sys , time as t , tomllib
from pathlib import Path
from modules import filters, ops_os , packet , payload_test_data as ptd , sdr

################
### SETTINGS ###
packet_size : str = 'S' # Available modes: 'S', 'M' or "L"
no_frames : int = 1 # Number of frames to transmit in one samples. WORKS ONLY FOR S and M frame sizes

dbg = True
plt = True
wrt = False

del_dst = False
################
################

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )
dst_dir = "test001.tx_rx_samples"
Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )
if del_dst :
    for file_path in Path ( dst_dir ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

if packet_size in ('S', 'M', 'L') :
    if dbg : print ( f"Running in {packet_size=}." )
else :
    raise ValueError ( f"Unknown {packet_size=}. Available modes: 'S', 'M' or 'L'." )

tx_samples = packet.TxSamples ()
match packet_size :
    case 'S' :
        #payload_bytes = ptd.PAYLOAD_1500BYTES_DEC
        payload_bytes = [ 15 ]
        while no_frames :
            tx_samples.add_frame ( payload_bytes = payload_bytes )
            no_frames -= 1
    case 'M' :
        while no_frames :
            payload_bytes = ptd.generate_payload_rand_up_2_1500b ()
            tx_samples.add_frame ( payload_bytes = payload_bytes )
            no_frames -= 1
    case 'L' :
        max_samples_size = int ( SAMPLES_BUFFER_SIZE * multiplicator ) # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)
        ptd.fill_samples_up_to_max_length ( tx_samples = tx_samples , max_samples_size = max_samples_size )

if dbg : print ( f"{tx_samples}" )
timestamp_group = ops_os.milis_timestamp ()
if wrt :
    tx_samples.save_symbols_from_samples_2_npf ( file_name = f"{timestamp_group}_tx_symbols_from_samples" , dir_name = dst_dir , add_timestamp = False )
    tx_samples.save_samples_2_npf ( file_name = f"{timestamp_group}_tx_samples" , dir_name = dst_dir , add_timestamp = False )
    tx_samples.save_active_samples_2_npf ( file_name = f"{timestamp_group}_tx_active_samples" , dir_name = dst_dir , add_timestamp = False )
if plt :
    tx_samples.plot_symbols_from_samples ( f"{script_filename} {timestamp_group}" )
    tx_samples.plot_samples ( f"{script_filename} {timestamp_group}" )
    tx_samples.plot_active_samples ( f"{script_filename} {timestamp_group}" )
    tx_samples.plot_samples_4_pluto_spectrum ( f"{script_filename} {timestamp_group}" )
