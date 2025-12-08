
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.6-tx.py
'''

import os
import time as t
import tomllib

#from pathlib import Path
from modules import ops_file , packet , sdr

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

pluto_rx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_RX"] )
for i in range ( 0 , 100 ) : # Clear buffer just to be safe
    raw_data = sdr.rx_samples ( pluto_rx )
print ( "Start Pluto Rx!" )
while True :
    rx_samples = sdr.rx_samples ( pluto_rx )
    rx_packets = packet.RxPackets ( samples = rx_samples )
    if rx_packets.has_sync :
        print ( "Preambuła znaleziona!" )
        break

'''
tx_packet = packet.TxPacket ( payload = settings[ "PAYLOAD_4BYTES_DEC" ] )
tx_packet.plot_symbols ( tx_packet.packet_symbols , script_filename + " BPSK packet symbols" )
tx_packet.plot_waveform ( tx_packet.payload_samples , script_filename + " BPSK payload waveform samples" , True)
tx_packet.plot_waveform ( tx_packet.packet_samples , script_filename + " BPSK packet waveform samples" )
tx_packet.plot_spectrum ( tx_packet.packet_samples , script_filename + " BPSK packet spectrum occupancy" )
'''

