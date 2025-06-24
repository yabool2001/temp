# 2025.06.22 Current priority:
# Split project for transmitting & receiving
# In receiver split thread for frames receiving and processing 
# This is receiving script 
'''
 Frame structure: [ preamble_bits , header_bits , payload_bits , crc32_bits ]
preamble_bit    [ 6 , 80 ]          2 bytes of fixed value preamble: 13 bits of BARKER 13 + 3 bits of padding
header_bits     [ X ]               1 byte of payload length = header value + 1
payload_bits    [ X , ... ]         variable length payload - max 256
crc32_bits      [ X , X , X , X ]   4 bytes of payload CRC32 
'''

import adi
import numpy as np
import os

from modules import filters , sdr , ops_packet , ops_file , modulation , corrections , plot
#from modules.rrc import rrc_filter
#from modules.clock_sync import polyphase_clock_sync

# App settings
verbose = True
verbose = False

# Inicjalizacja plików CSV
csv_filename_tx_waveform = "complex_tx_waveform.csv"
csv_filename_rx_waveform = "complex_rx_waveform.csv"
csv_filename_tx_symbols = "complex_tx_symbols.csv"
csv_filename_rx_symbols = "complex_rx_symbols.csv"
csv_filename_corr_and_filtered_rx_samples = "corr_and_filtered_rx_samples.csv"
csv_filename_aligned_rx_samples = "aligned_rx_samples.csv"

script_filename = os.path.basename ( __file__ )

# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = 2900e6     # częstotliwość nośna [Hz]
#F_S = 2e6     # częstotliwość próbkowania [Hz] >= 521e3 && <
BW  = 1_000_000         # szerokość pasma [Hz]
#F_S = 521100     # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100
print (f"{F_S=}")
SPS = 4                 # próbek na symbol
TX_GAIN = -10.0
URI = "ip:192.168.2.1"
#URI = "usb:"

RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach
CYCLE_MS = 10           # opóźnienie między pakietami [ms]; <0 = liczba powtórzeń

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie


# ------------------------ KONFIGURACJA SDR ------------------------
def main():
    packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
    print ( f"{packet_bits=}" )
    tx_bpsk_symbols = modulation.create_bpsk_symbols ( packet_bits )
    #plot.plot_bpsk_symbols ( tx_bpsk_symbols , script_filename + " tx_bpsk_symbols" )
    print ( f"{tx_bpsk_symbols=}" )
    tx_samples = filters.apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )
    #plot.plot_complex_waveform ( tx_samples , script_filename + " tx_samples")
    pluto = sdr.init_pluto ( URI , F_C , F_S , BW )
    if verbose : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    sdr.tx_cyclic ( tx_samples , pluto )

    # Clear buffer just to be safe
    for i in range ( 0 , 10 ) :
        raw_data = sdr.rx_samples ( pluto )
    # Receive samples
    rx_samples = sdr.rx_samples ( pluto )
    sdr.stop_tx_cyclic ( pluto )
    #plot.plot_complex_waveform ( rx_samples , script_filename + " rx_samples" )
    preamble_symbols = modulation.create_bpsk_symbols ( ops_packet.BARKER13 )
    preamble_samples = filters.apply_tx_rrc_filter ( preamble_symbols , SPS , RRC_BETA , RRC_SPAN , True )
    #rx_samples_filtered = filters.apply_rrc_rx_filter ( rx_samples , SPS , RRC_BETA , RRC_SPAN , False ) # W przyszłości rozważyć implementację tego filtrowania sampli rx
    rx_samples_phase_corrected = corrections.phase_shift_corr ( rx_samples )
    #plot.plot_complex_waveform ( rx_samples_phase_corrected , script_filename + " rx_samples_phase_corrected" )
    corr_and_filtered_rx_samples = filters.apply_tx_rrc_filter ( rx_samples_phase_corrected , SPS , RRC_BETA , RRC_SPAN , upsample = False ) # Może zmienić na apply_rrc_rx_filter
    print ( f"{corr_and_filtered_rx_samples.size=}")
    while ( corr_and_filtered_rx_samples.size > 0 ) :
        corr = np.correlate ( corr_and_filtered_rx_samples , preamble_samples , mode = 'full' )
        peak_index = np.argmax ( np.abs ( corr ) )
        timing_offset = peak_index - len ( preamble_samples ) + 1
        print ( f"{timing_offset=} | ")
        aligned_rx_samples = corr_and_filtered_rx_samples[ timing_offset: ]
        print ( f"{aligned_rx_samples.size=}")
        #plot.plot_complex_waveform ( aligned_rx_samples , script_filename + " aligned_rx_samples" )
        if ops_packet.is_preamble ( aligned_rx_samples , RRC_SPAN , SPS ) :
            try: # Wstawiłem to 24.06.2025, żeby rozkminić błąd TypeError: cannot unpack non-iterable NoneType object i nie wiem czy się sprawdzic
                payload_bits , clip_samples_index = ops_packet.get_payload_bytes ( aligned_rx_samples , RRC_SPAN , SPS )
            except :
                pass
            if payload_bits is not None and clip_samples_index is not None :
                print ( f"{payload_bits=}" )
                corr_and_filtered_rx_samples = aligned_rx_samples[ int ( clip_samples_index ) ::]
                print ( f"{corr_and_filtered_rx_samples.size=}")
            else :
                print ( "No payload. Leftovers saved to add to next samples. Breaking!" )
                leftovers = corr_and_filtered_rx_samples
                break
        else :
            print ( "No preamble. Leftovers saved to add to next samples. Breaking!" )
            leftovers = corr_and_filtered_rx_samples
            break
        print ( f"{timing_offset=} | ")

    acg_vaule = pluto._get_iio_attr ( 'voltage0' , 'hardwaregain' , False )
    # Stop transmitting

    csv_corr_and_filtered_rx_samples , csv_writer_corr_and_filtered_rx_samples = ops_file.open_and_write_samples_2_csv ( csv_filename_corr_and_filtered_rx_samples , corr_and_filtered_rx_samples )
    csv_aligned_rx_samples , csv_writer_aligned_rx_samples = ops_file.open_and_write_samples_2_csv ( csv_filename_aligned_rx_samples , aligned_rx_samples )
    ops_file.flush_data_and_close_csv ( csv_corr_and_filtered_rx_samples )
    ops_file.flush_data_and_close_csv ( csv_aligned_rx_samples )
    ops_file.plot_samples ( csv_filename_corr_and_filtered_rx_samples )
    ops_file.plot_samples ( csv_filename_aligned_rx_samples )

    print ( f"{acg_vaule=}" )

if __name__ == "__main__":
    main ()
