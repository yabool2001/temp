
'''

Skrypt będzie służył to porównywania offsetów między aktywnymi symbolami, a aktywnymi próbkami, przy różnej wartości miejsca pierwszego symbolu.
Do tego testu posłuży się funkcji offsets_accuracy_test() z klasy TxSamples, która porówna 

'''

import numpy as np , os , socket , sys , time as t , tomllib
from pathlib import Path
from modules import ops_os , packet , payload_test_data as ptd , sdr

################
### SETTINGS ###

mode : str = 'inference' # Available modes: 'training', 'test' or "inference"
ASCII_FRAME_SIZE : bytes = b"S" # Available frame sizes: b"S", b"M" or b"L" (small, medium or large)

dbg = True
plt = False
wrt = True
del_dst = True

################
################

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

dst_dir = f"np.{mode}"
Path ( dst_dir ).mkdir ( parents = True , exist_ok = True )
if del_dst :
    for file_path in Path ( dst_dir ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

if mode in ('training', 'test', 'inference') :
    if dbg : print ( f"Running in {mode=}." )
else :
    raise ValueError ( f"Unknown {mode=}. Available modes: 'training', 'test' or 'inference'." )

SAMPLES_BUFFER_SIZE_MULTIPLICATOR = 2
SAMPLES_BUFFER_SIZE = int ( toml_settings["ADALM-Pluto"][ "SAMPLES_BUFFER_SIZE" ] )

def wrt_flat_tensor ( tx_samples : packet.TxSamples , timestamp_group : str ) -> None :
    tx_samples.save_active_symbols_2_npf ( file_name = f"{timestamp_group}_tx_active_symbols" , dir_name = dst_dir , add_timestamp = False ) # wrersja bez rozbiegówki i wygaszenia, czyli tylko aktywne próbki z ramek, bez 0+j0
    #tx_samples.save_samples_4_pluto_2_npf ( file_name = f"{timestamp_group}_tx_samples4pluto" , dir_name = dst_dir , add_timestamp = False )
    tx_samples.save_samples_2_npf ( file_name = f"{timestamp_group}_tx_samples" , dir_name = dst_dir , add_timestamp = False )
    if dbg : print ( f"Samples and corresponding symbols saved to npf type files in {dst_dir=} {timestamp_group=}..." )

def build_tx_samples_and_timestamp_group ( multiplicator : float = SAMPLES_BUFFER_SIZE_MULTIPLICATOR , frame_size : bytes = b"S" ) -> tuple [ packet.TxSamples , str ] :
    tx_samples = packet.TxSamples ()
    match frame_size :
        case b"S" :
            #payload_bytes = ptd.PAYLOAD_1500BYTES_DEC
            payload_bytes = [ 15 ]
            tx_samples.add_frame ( payload_bytes = payload_bytes )
        case b"M" :
            payload_bytes = ptd.generate_payload_rand_up_2_1500b ()
            tx_samples.add_frame ( payload_bytes = payload_bytes )
        case b"L" :
            max_samples_size = int ( SAMPLES_BUFFER_SIZE * multiplicator ) # Maksymalna liczba próbek do wysłania w jednej transmisji (80% bufora, aby zostawić miejsce na rozpędzenie się filtra)
            ptd.fill_samples_up_to_max_length ( tx_samples = tx_samples , max_samples_size = max_samples_size )
    if dbg : print ( f"{tx_samples.samples.size=}" )
    timestamp_group = ops_os.milis_timestamp ()
    if wrt : wrt_flat_tensor ( tx_samples = tx_samples , timestamp_group = timestamp_group )
    return tx_samples , timestamp_group

tx_samples , timestamp_group = build_tx_samples_and_timestamp_group ( multiplicator = SAMPLES_BUFFER_SIZE_MULTIPLICATOR , frame_size = ASCII_FRAME_SIZE )
if plt :
    tx_samples.plot_active_symbols ( f"{script_filename} {timestamp_group}" )
    tx_samples.plot_active_samples ( f"{script_filename} {timestamp_group}" )
    tx_samples.plot_samples ( f"{script_filename} {timestamp_group}" )
tx_samples.offsets_accuracy_test ()