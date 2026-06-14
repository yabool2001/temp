# Skrypt do testowanie klasy RxSamples, RxFrame i RxSymbol, które służą do odbierania i demodulowania próbek sygnału.

from modules import ops_file, ops_os , packet, plot , sdr
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import os , sys , tomllib
import socket
import time as t
import tomllib

################
### SETTINGS ###
MODE : str = 'test' # Available modes: 'training', 'test' or "inference"
SYMBOLS_SRC : str = "active_samples"

dbg = True
plt = False
wrt = True
del_dir = False

################
################

np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy
script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

filename = '_rx_samples_'
dir_name = "test001.tx_rx_samples"
Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
if del_dir :
    for file_path in Path ( dir_name ).glob ( "*" ) :
        if file_path.is_file () :
            file_path.unlink ( missing_ok = True )

if MODE in ('training', 'test', 'inference') :
    if dbg : print ( f"Running in {MODE=}." )
else :
    raise ValueError ( f"Unknown {MODE=}. Available modes: 'training', 'test' or 'inference'." )
timestamp_groups = sorted ( { p.name.split ( "_rx_samples_" , 1 )[ 0 ] for p in Path ( dir_name ).glob("*_rx_samples_*.npy") } )
rx_samples_files = sorted ( Path ( dir_name ).glob ( f"{timestamp_groups[0]}{filename}*.npy" ) )
rx_samples = packet.RxSamples ()
for samples_file in rx_samples_files :
    rx_samples.rx ( filename_and_dirname = str ( samples_file ) , concatenate = True )
#if plt : plot.complex_waveform_v0_1_6 ( rx_samples.create_corr_seq_samples ( clip_tail = True ) , f"Sync sequence samples (clipped) " )
if plt : rx_samples.plot_samples ( title = f"{script_filename}" , mark_samples = True )
rx_samples.detect_frames ( deep = False , samples_filtered = False , correct_samples = False , add_peak_at_0 = False )
first_sample_idx = rx_samples.create_X_train_samples_and_y_train_tensor ( src_dir = Path ( dir_name ) , timestamp_group = timestamp_groups[0] , X_train_samples_filtered = False , symbols_src = SYMBOLS_SRC )
if first_sample_idx != rx_samples.first_symbol_idx : print ( f"WARNING! WARNING! WARNING!: {first_sample_idx=} != {rx_samples.first_symbol_idx=}" )
if plt :
    rx_samples.plot_samples ( title = f"{script_filename}" , mark_samples = True )
    rx_samples.plot_X_and_y ( title = f"{script_filename}" , mark_samples = True )
if wrt : rx_samples.save_X_and_y ( dir_name = dir_name , timestamp_group = timestamp_groups[0] )

if dbg : print ( f"\n{script_filename=} {rx_samples=}" )