# 2025.06.22 Current priority:
# Split project for transmitting & receiving
# In receiver split thread for frames receiving and processing 
# This is receiving script 
# Sygnał odebrany (sample) jest wartością zespoloną, której reprezentacja zawiera informację o amplitudzie i fazie. W praktyce, zwłaszcza w komunikacji radiowej, sygnał odbierany może mieć zmienną fazę wynikającą z różnicy częstotliwości lokalnych oscylatorów (LO) nadajnika i odbiornika oraz dryftów częstotliwości.
'''
 Frame structure: [ preamble_bits , header_bits , payload_bits , crc32_bits ]
preamble_bit    [ 6 , 80 ]          2 bytes of fixed value preamble: 13 bits of BARKER 13 + 3 bits of padding
header_bits     [ X ]               1 byte of payload length = header value + 1
payload_bits    [ X , ... ]         variable length payload - max 256
crc32_bits      [ X , X , X , X ]   4 bytes of payload CRC32 
'''

import adi
import json
import keyboard
import numpy as np
import os
import time as t

from modules import filters , sdr , ops_packet , ops_file , modulation , monitor , corrections , plot
#from modules.rrc import rrc_filter
#from modules.clock_sync import polyphase_clock_sync

# Wczytaj plik JSON z konfiguracją
with open ( "settings.json" , "r" ) as settings_file :
    settings = json.load ( settings_file )

### App settings ###
real_rx = True  # Pobieranie żywych danych z Pluto 
#real_rx = False # Ładowanie danych zapisanych w pliku:

#rx_saved_filename = "logs/rx_samples_10k.csv"
#rx_saved_filename = "logs/rx_samples_32768.csv"
#rx_saved_filename = "logs/rx_samples_1255-barely_payload.csv"
rx_saved_filename = "logs/rx_samples_1240-no_payload.csv"
#rx_saved_filename = "logs/rx_samples_987-no_crc32.csv"
#rx_saved_filename = "logs/rx_samples_702-no_preamble.csv"
#rx_saved_filename = "logs/rx_samples_1245-no_barker.csv"

script_filename = os.path.basename ( __file__ )

# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = int ( settings["ADALM-Pluto"]["F_C"] )    # częstotliwość nośna [Hz]
BW  = int ( settings["ADALM-Pluto"]["BW"] )        # szerokość pasma [Hz]
#F_S = 521100     # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = int ( BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100 )
SPS = int ( settings["bpsk"]["SPS"] )                # próbek na symbol
TX_GAIN = float ( settings["ADALM-Pluto"]["TX_GAIN"] )
URI = settings["ADALM-Pluto"]["URI"]["IP"]
#URI = settings["ADALM-Pluto"]["URI"]["USB"]"

RRC_BETA = float ( settings["rrc_filter"]["BETA"] )    # roll-off factor
RRC_SPAN = int ( settings["rrc_filter"]["SPAN"] )    # długość filtru RRC w symbolach

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie
if settings["log"]["verbose_2"] : print (f"{F_C=} {F_S=} {BW=} {SPS=} {RRC_BETA=} {RRC_SPAN=}")
test = settings["log"]["verbose_0"]

# ------------------------ KONFIGURACJA SDR ------------------------
def main() :

    packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
    if settings["log"]["verbose_2"] : print ( f"{packet_bits=}" )
    tx_bpsk_symbols = modulation.create_bpsk_symbols ( packet_bits )
    if settings["log"]["verbose_1"] : print ( f"{tx_bpsk_symbols=}" )
    tx_samples = filters.apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )

    uri_tx = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
    #uri_rx = sdr.get_uri ( "10447318ac0f00091e002400454e18b77d" , "usb" )
    #uri_tx = sdr.get_uri ( "10447318ac0f00091e002400454e18b77d" , "usb" )
    #uri_rx = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
    pluto_tx = sdr.init_pluto ( uri_tx , settings["ADALM-Pluto"]["F_C"] , F_S , BW )
    #pluto_rx = sdr.init_pluto ( uri_tx , settings["ADALM-Pluto"]["F_C"] , F_S , BW )
    if settings["log"]["verbose_1"] : print ( f"{uri_tx=}" )
    if settings["log"]["verbose_0"] : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    sdr.stop_tx_cyclic ( pluto_tx )
    while True :
        print ( "Naciśnij klawisz 't', aby wysłać pakiet.")
        keyboard.wait ( "t" )
        packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
        tx_bpsk_symbols = modulation.create_bpsk_symbols ( packet_bits )
        #if settings["log"]["verbose_0"] : plot.plot_bpsk_symbols ( tx_bpsk_symbols , script_filename + " tx_bpsk_symbols" )
        if settings["log"]["verbose_0"] : print ( f"{tx_bpsk_symbols=}" )
        tx_samples = filters.apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )
        if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_samples , script_filename + f" {tx_samples.size=}" )
        sdr.tx_once ( tx_samples , pluto_tx )
        if settings["log"]["verbose_1"] : print ( f"{packet_bits=} sent" )

if __name__ == "__main__":
    main ()
