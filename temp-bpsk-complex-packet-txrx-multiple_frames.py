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
import numpy as np
import os

from modules import filters , sdr , ops_packet , ops_file , modulation , monitor , corrections , plot
#from modules.rrc import rrc_filter
#from modules.clock_sync import polyphase_clock_sync

# Wczytaj plik JSON z konfiguracją
with open ( "settings.json" , "r" ) as settings_file :
    settings = json.load ( settings_file )

### App settings ###
#real_rx = True  # Pobieranie żywych danych z Pluto 
real_rx = False # Ładowanie danych zapisanych w pliku:

rx_saved_filename = "logs/rx_samples_10k.csv"

script_filename = os.path.basename ( __file__ )

# ------------------------ PARAMETRY KONFIGURACJI ------------------------
F_C = 820e6     # częstotliwość nośna [Hz]
#F_S = 2e6     # częstotliwość próbkowania [Hz] >= 521e3 && <
BW  = 1_000_000         # szerokość pasma [Hz]
#F_S = 521100     # częstotliwość próbkowania [Hz] >= 521e3 && <
F_S = BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100
SPS = 4                 # próbek na symbol
TX_GAIN = -10.0
URI = "ip:192.168.2.1"
#URI = "usb:"

RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # można zmieniać dynamicznie
if settings["log"]["verbose_2"] : print (f"{F_C=} {F_S=} {BW=} {SPS=} {RRC_BETA=} {RRC_SPAN=}")
test = settings["log"]["verbose_0"]

# ------------------------ KONFIGURACJA SDR ------------------------
def main() :

    packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
    if settings["log"]["verbose_2"] : print ( f"{packet_bits=}" )
    tx_bpsk_symbols = modulation.create_bpsk_symbols ( packet_bits )
    if settings["log"]["verbose_2"] : print ( f"{tx_bpsk_symbols=}" )
    tx_samples = filters.apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )

    if real_rx :
        uri_tx = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
        uri_rx = sdr.get_uri ( "10447318ac0f00091e002400454e18b77d" , "usb" )
        #uri_tx = sdr.get_uri ( "10447318ac0f00091e002400454e18b77d" , "usb" )
        #uri_rx = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
        pluto_tx = sdr.init_pluto ( uri_tx , settings["ADALM-Pluto"]["F_C"] , F_S , settings["ADALM-Pluto"]["BW"] )
        pluto_rx = sdr.init_pluto ( uri_rx , settings["ADALM-Pluto"]["F_C"] , F_S , settings["ADALM-Pluto"]["BW"] )
        if settings["log"]["verbose_0"] : print ( f"{uri_tx=}" ) ; print ( f"{uri_rx=}" )
        if settings["log"]["verbose_0"] : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
        sdr.tx_cyclic ( tx_samples , pluto_tx )

        # Clear buffer just to be safe
        for i in range ( 0 , 10 ) :
            raw_data = sdr.rx_samples ( pluto_rx )
            if settings["log"]["verbose_0"] : monitor.plot_fft_p2 ( raw_data , F_S )
        # Receive samples
        rx_samples = sdr.rx_samples ( pluto_rx )
        sdr.stop_tx_cyclic ( pluto_tx )
    else :
        rx_samples = ops_file.open_csv_and_load_np_complex128 ( rx_saved_filename )
    
    preamble_samples = modulation.get_barker13_bpsk_samples ( SPS , RRC_BETA , RRC_SPAN , True )
    rx_samples_filtered = filters.apply_rrc_rx_filter ( rx_samples , SPS , RRC_BETA , RRC_SPAN , False )
    #rx_samples_simple_correlated = corrections.simple_correlation ( rx_samples_filtered , modulation.get_barker13_bpsk_samples ( SPS , RRC_BETA , RRC_SPAN ) )
    #plot.plot_complex_waveform ( rx_samples_simple_correlated , script_filename + f" rx_samples_simple_correlated , {rx_samples_simple_correlated.size=}" )
    #rx_samples_corrected = corrections.full_compensation ( rx_samples_simple_correlated , F_S , modulation.get_barker13_bpsk_samples ( SPS , RRC_BETA , RRC_SPAN ) )
    #rx_samples_corrected = corrections.costas_loop ( rx_samples_filtered , F_S )
    rx_samples_corrected = corrections.full_compensation ( rx_samples_filtered , F_S , modulation.get_barker13_bpsk_samples ( SPS , RRC_BETA , RRC_SPAN , True ) )
    rx_samples_corrected = modulation.zero_quadrature ( rx_samples_corrected )
    rx_samples_corrected_temp = rx_samples_corrected.copy ()
    if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( rx_samples_corrected_temp , script_filename + f" {rx_samples_corrected_temp.size=}" )
    #corr_and_filtered_rx_samples = filters.apply_tx_rrc_filter ( rx_samples_corrected , SPS , RRC_BETA , RRC_SPAN , upsample = False ) # Może zmienić na apply_rrc_rx_filter
    print ( f"{rx_samples_corrected.size=} ")
    counter = 0
    while ( rx_samples_corrected.size > 0 ) :
        counter += 1
        if settings["log"]["verbose_1"] : print ( f"{counter=}" )
        corr = np.correlate ( rx_samples_corrected , preamble_samples , mode = 'full' )
        peak_index = np.argmax (  corr  )
        #mean_corr = np.mean ( np.abs ( corr ) )
        #std_corr = np.std ( np.abs ( corr ) )
        #threshold = mean_corr + 3 * std_corr
        threshold = 0.99 * peak_index
        detected_peaks = np.where ( corr  >= threshold ) [0]
        first_index = detected_peaks[0]
        timing_offset = first_index - len ( preamble_samples ) + 1
        rx_samples_aligned = rx_samples_corrected[ timing_offset: ]
        if settings["log"]["verbose_1"] : print ( f"{timing_offset=} | {rx_samples_aligned.size=}")
        #plot.plot_complex_waveform ( rx_samples_aligned , script_filename + " rx_samples_aligned" )
        if ops_packet.is_preamble ( rx_samples_aligned , RRC_SPAN , SPS ) :
            try: # Wstawiłem to 24.06.2025, żeby rozkminić błąd TypeError: cannot unpack non-iterable NoneType object i nie wiem czy się sprawdzic
                payload_bits , clip_samples_index = ops_packet.get_payload_bytes ( rx_samples_aligned , RRC_SPAN , SPS )
            except :
                pass
            if payload_bits is not None and clip_samples_index is not None :
                if settings["log"]["verbose_1"] : print ( f"{payload_bits=}" )
                rx_samples_corrected = rx_samples_aligned[ int ( clip_samples_index ) ::]
                if settings["log"]["verbose_1"] : print ( f"{rx_samples_corrected.size=}")
            else :
                if settings["log"]["verbose_2"] : print ( "No payload. Leftovers saved to add to next samples. Breaking!" )
                leftovers = rx_samples_corrected
                break
        else :
            if settings["log"]["verbose_2"] : print ( "No preamble. Leftovers saved to add to next samples. Breaking!" )
            leftovers = rx_samples_corrected
            break

    if settings["log"]["verbose_2"] and real_rx : acg_vaule = pluto_rx._get_iio_attr ( 'voltage0' , 'hardwaregain' , False ) ; print ( f"{acg_vaule=}" )
    # Stop transmitting

    corrections.estimate_cfo_drit ( rx_samples , F_S )
    corrections.estimate_cfo_drit ( rx_samples_corrected_temp , F_S )

    plot.plot_bpsk_symbols ( tx_bpsk_symbols , script_filename + f" {tx_bpsk_symbols.size=}" )
    if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( tx_samples , script_filename + f" {tx_samples.size=}" )
    if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( preamble_samples , script_filename + f" {preamble_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples , script_filename + f" {rx_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples_filtered , script_filename + f" {rx_samples_filtered.size=}" )
    if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( rx_samples_aligned , script_filename + f" {rx_samples_aligned.size=}" )

    if real_rx :
        ops_file.write_samples_2_csv ( settings["log"]["tx_samples"] , tx_samples )
        csv_rx_samples , csv_writer_rx_samples = ops_file.open_and_write_samples_2_csv ( settings["log"]["rx_samples"] , rx_samples )
        csv_rx_samples_filtered , csv_writer_rx_samples_filtered = ops_file.open_and_write_samples_2_csv ( settings["log"]["rx_samples_filtered"] , rx_samples_filtered )
        ops_file.write_samples_2_csv ( settings["log"]["rx_samples_corrected"] , rx_samples_corrected_temp )
        csv_rx_samples_aligned , csv_writer_rx_samples_aligned = ops_file.open_and_write_samples_2_csv ( settings["log"]["rx_samples_aligned"] , rx_samples_aligned )
        ops_file.flush_data_and_close_csv ( csv_rx_samples )
        ops_file.flush_data_and_close_csv ( csv_rx_samples_filtered )
        ops_file.flush_data_and_close_csv ( csv_rx_samples_aligned )

if __name__ == "__main__":
    main ()
