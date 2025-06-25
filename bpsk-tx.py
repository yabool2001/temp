# 2025.06.22 Current priority:
# Split project for transmitting & receiving
# In receiver split thread for frames receiving and processing 
# This is a script for transmitting frames
'''
Frame structure: [ preamble_bits , header_bits , payload_bits , crc32_bits ]
preamble_bit    [ 6 , 80 ]          2 bytes of fixed value preamble: 13 bits of BARKER 13 + 3 bits of padding
header_bits     [ X ]               1 byte of payload length = header value + 1
payload_bits    [ X , ... ]         variable length payload - max 256
crc32_bits      [ X , X , X , X ]   4 bytes of payload CRC32 
'''

import json
import numpy as np
import os
import time

from modules import filters , sdr , ops_packet , ops_file , modulation , corrections , plot

# Wczytaj plik JSON z konfiguracją
with open ( "settings.json" , "r" ) as settings_file :
    settings = json.load ( settings_file )

# App settings
#verbose = True
verbose = False

# Inicjalizacja plików CSV
csv_filename_tx_waveform = "complex_tx_waveform.csv"
csv_filename_tx_symbols = "complex_tx_symbols.csv"

script_filename = os.path.basename ( __file__ )

# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_S = settings["ADALM-Pluto"]["BW"] * 3 if ( settings["ADALM-Pluto"]["BW"] * 3 ) >= 521100 and ( settings["ADALM-Pluto"]["BW"] * 3 ) <= 61440000 else 521100
print (f"{F_S=}")
SPS = 4                 # próbek na symbol
TX_GAIN = -1.0
#URI = "ip:192.168.2.1"
URI = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
print ( f"Tx {URI=}" )

RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach
CYCLE_MS = 10           # opóźnienie między pakietami [ms]; <0 = liczba powtórzeń

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie


# ------------------------ KONFIGURACJA SDR ------------------------
def main():
    packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
    tx_symbols = modulation.create_bpsk_symbols ( packet_bits )
    #plot.plot_bpsk_symbols ( tx_bpsk_symbols , script_filename + " tx_bpsk_symbols" )
    print ( f"{tx_symbols=}" )
    tx_samples = filters.apply_tx_rrc_filter ( tx_symbols , settings["bpsk"]["SPS"] , settings["rrc_filter"]["BETA"] , settings["rrc_filter"]["SPAN"] , True )
    #plot.plot_complex_waveform ( tx_samples , script_filename + " tx_samples")
    pluto = sdr.init_pluto_v2 ( URI , settings["ADALM-Pluto"]["F_C"] , F_S , settings["ADALM-Pluto"]["BW"] , TX_GAIN )
    # if verbose : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    sdr.tx_cyclic ( tx_samples , pluto )
    print ( f"Start Tx cyclic {packet_bits=}" )

    # Clear buffer just to be safe
    for i in range ( 0 , 10 ) :
        raw_data = sdr.rx_samples ( pluto )
    
    try :
        print("Program działa... Naciśnij Ctrl+C, aby zakończyć.")
        while True :
            time.sleep ( 1 )
    except KeyboardInterrupt :
        print ( "Zakończono ręcznie (Ctrl+C)" )
    finally:
        sdr.tx_destroy_buffer()
        sdr.tx_cyclic_buffer = False
        print ( f"{sdr.tx_cyclic_buffer=}" )

    csv_tx_samples , csv_writer_tx_samples = ops_file.open_and_write_samples_2_csv ( settings["log"]["tx_samples"] , tx_samples )
    ops_file.flush_data_and_close_csv ( csv_tx_samples )
    if verbose : ops_file.plot_samples ( settings["log"]["tx_samples"] ) , ops_file.plot_samples ( settings["log"]["tx_symbols"] )

if __name__ == "__main__":
    main ()
