import adi
import csv
import numpy as np
from scipy.signal import lfilter
import time
import zlib
from scipy.signal import upfirdn
import pandas as pd
import plotly.express as px

from modules.rrc import rrc_filter
from modules.clock_sync import polyphase_clock_sync

# Inicjalizacja pliku CSV
csv_filename_raw = "complex_rx_raw.csv"
csv_filename_filtered = "complex_rx_filtered.csv"
csv_file_raw = open ( csv_filename_raw , mode = "w" , newline = '' )
csv_file_filtered = open ( csv_filename_filtered , mode = "w" , newline = '' )
csv_writer_raw = csv.writer ( csv_file_raw )
csv_writer_filtered = csv.writer ( csv_file_filtered )
csv_writer_raw.writerow ( [ "timestamp" , "real" , "imag" ] )
csv_writer_filtered.writerow ( [ "timestamp" , "real" , "imag" ] )


# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = 2_900_000_000     # częstotliwość nośna [Hz]
#F_S = 521_100          # częstotliwość próbkowania [Hz]
F_S = 3_000_000         # częstotliwość próbkowania [Hz]
SPS = 4                 # próbek na symbol
TX_ATTENUATION = 10.0   # tłumienie TX [dB]
BW = 1_000_000          # szerokość pasma [Hz]
RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach
CYCLE_MS = 10           # opóźnienie między pakietami [ms]; <0 = liczba powtórzeń

# ------------------------ DANE DO MODULACJI ------------------------
header = [ 0xAA , 0xAA , 0xAA , 0xAA ]
payload = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie

def create_packet(header, payload):
    length_byte = [len(payload) - 1]
    crc32 = zlib.crc32(bytes(payload))
    crc_bytes = list(crc32.to_bytes(4, 'big'))
    return header + length_byte + payload + crc_bytes

def bits_to_bpsk(bits):
    return np.array([1.0 if bit else -1.0 for bit in bits], dtype=np.float32)

def rrc_filter(beta, sps, span):
    N = sps * span
    t = np.arange(-N//2, N//2 + 1, dtype=np.float32) / sps
    taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    taps[np.isnan(taps)] = 0
    taps /= np.sqrt(np.sum(taps**2))
    return taps

'''
def modulate_packet(packet, sps, beta, span):
    bits = np.unpackbits(np.array(packet, dtype=np.uint8))
    symbols = bits_to_bpsk(bits)
    rrc = rrc_filter(beta, sps, span)
    shaped = upfirdn(rrc, symbols, up=sps)
    return (shaped + 0j).astype(np.complex64)
'''

def modulate_packet_2 ( packet , sps , beta , span ) :
    bits = np.unpackbits(np.array(packet, dtype=np.uint8))
    symbols = bits_to_bpsk(bits)
    shaped = upfirdn(rrc, symbols, up=sps)
    rrc = rrc_filter(beta, sps, span)
    return upfirdn(rrc, symbols, up=sps)

def shape ( shaped ):
    return (shaped + 0j).astype(np.complex64)

def write_waveform ( waveform , file, writer , t0 ) :
    for sample in waveform :
        writer.writerow ( [ time.time () - t0 , sample.real , sample.imag ] )
    file.flush ()
        
# ------------------------ KONFIGURACJA SDR ------------------------
def main():
    packet = create_packet ( header , payload )
    print (packet )
    # waveform = modulate_packet ( packet , SPS , RRC_BETA , RRC_SPAN )
    waveform = modulate_packet_2 ( packet , SPS , RRC_BETA , RRC_SPAN )
    waveform = shape (waveform)
    t0 = time.time ()
    write_waveform ( waveform , csv_file_filtered , csv_writer_filtered , t0 )
    time.sleep ( CYCLE_MS / 1000.0 )
    write_waveform ( waveform , csv_file_filtered , csv_writer_filtered , t0 )

if __name__ == "__main__":
    main ()
