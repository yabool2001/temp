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
    if settings["log"]["verbose_2"] : print ( f"{tx_bpsk_symbols=}" )
    tx_samples = filters.apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )

    #uri_tx = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
    uri_rx = sdr.get_uri ( "10447318ac0f00091e002400454e18b77d" , "usb" )
    #uri_tx = sdr.get_uri ( "10447318ac0f00091e002400454e18b77d" , "usb" )
    #uri_rx = sdr.get_uri ( "1044739a470b000a090018007ecf7f5ea8" , "usb" )
    #pluto_tx = sdr.init_pluto ( uri_tx , settings["ADALM-Pluto"]["F_C"] , F_S , BW )
    pluto_rx = sdr.init_pluto ( uri_rx , settings["ADALM-Pluto"]["F_C"] , F_S , BW )
    if settings["log"]["verbose_0"] : print ( f"{uri_rx=}" )
    if settings["log"]["verbose_0"] : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    #sdr.tx_cyclic ( tx_samples , pluto_tx )
    # Clear buffer just to be safe
    for i in range ( 0 , 10 ) :
        raw_data = sdr.rx_samples ( pluto_rx )
        if settings["log"]["verbose_0"] : monitor.plot_fft_p2 ( raw_data , F_S )
    # Receive and process samples
    barker13_samples = modulation.get_barker13_bpsk_samples ( SPS , RRC_BETA , RRC_SPAN , True )
    print ( "Start Rx!" ) 
    while True :
        rx_samples = sdr.rx_samples ( pluto_rx )
        #if ops_packet.is_sync_seq ( filters.apply_rrc_rx_filter ( rx_samples , SPS , RRC_BETA , RRC_SPAN , False ) , barker13_samples ) :
        if ops_packet.is_sync_seq ( rx_samples ,barker13_samples ) :
            print ( "Yes!" ) 
            #sdr.stop_tx_cyclic ( pluto_tx )
            start = t.perf_counter ()
            rx_samples_filtered = filters.apply_rrc_rx_filter ( rx_samples , SPS , RRC_BETA , RRC_SPAN , False )
            end = t.perf_counter ()
            print ( f"apply_rrc_rx_filter perf: {end - start:.6f} sekundy" )
            start = t.perf_counter ()
            rx_samples_corrected = corrections.full_compensation ( rx_samples_filtered , F_S , modulation.get_barker13_bpsk_samples ( SPS , RRC_BETA , RRC_SPAN , True ) )
            end = t.perf_counter ()
            print ( f"full_compensation perf: {end - start:.6f} sekundy" )
            start = t.perf_counter ()
            rx_samples_corrected = modulation.zero_quadrature ( rx_samples_corrected )
            end = t.perf_counter ()
            print ( f"zero_quadrature perf: {end - start:.6f} sekundy" )
            if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( rx_samples_corrected , script_filename + f" {rx_samples_corrected.size=}" )
            #corr_and_filtered_rx_samples = filters.apply_tx_rrc_filter ( rx_samples_corrected , SPS , RRC_BETA , RRC_SPAN , upsample = False ) # Może zmienić na apply_rrc_rx_filter
            if settings["log"]["verbose_1"] : print ( f"{rx_samples_corrected.size=} ")
            start = t.perf_counter ()
            #corr = modulation.normalized_cross_correlation ( rx_samples_corrected , barker13_samples )
            #corr = np.correlate ( rx_samples_corrected , barker13_samples , mode = 'full' )
            corr = modulation.fast_normalized_cross_correlation ( rx_samples_corrected , barker13_samples )
            #corr = modulation.fft_normalized_cross_correlation ( rx_samples_corrected , barker13_samples )
            end = t.perf_counter ()
            print ( f"corr perf: {end - start:.6f} sekundy" )
            threshold = np.mean ( corr ) + 3 * np.std ( corr )
            #threshold = 0.7  # ustalony eksperymentalnie, np. 0.5–0.8 i wymaga znormalizowanych danych
            #threshold = np.percentile ( corr, 99.5 ) # lub adaptacyjnie
            detected_peaks = np.where(corr >= threshold)[0]
            peaks = modulation.group_peaks_by_distance ( detected_peaks , corr , min_distance = 2 )
            if settings["log"]["verbose_1"] : print ( f"\n{peaks.size=} | {rx_samples_corrected.size=}")
            for peak in peaks :
                #rx_samples_corrected = rx_samples_corrected[ peak: ]
                #plot.plot_complex_waveform ( rx_samples_corrected , script_filename + " rx_samples_corrected" )
                if ops_packet.is_preamble ( rx_samples_corrected[ peak: ] , RRC_SPAN , SPS ) :
                    try: # Wstawiłem to 24.06.2025, żeby rozkminić błąd TypeError: cannot unpack non-iterable NoneType object i nie wiem czy się sprawdzic
                        payload_bits = ops_packet.get_payload_bytes ( rx_samples_corrected[ peak: ] , RRC_SPAN , SPS )
                    except :
                        pass
                    if payload_bits is not None :
                        if settings["log"]["verbose_1"] : print ( f"{payload_bits=} {rx_samples_corrected.size=}, {peak=}" )
                        #rx_samples_corrected = rx_samples_corrected[ int ( clip_samples_index ) ::]
                    else :
                        if settings["log"]["verbose_2"] : print ( "No payload. Leftovers saved to add to next samples. Breaking!" )
                else :
                    if settings["log"]["verbose_2"] : print ( "No preamble. Leftovers saved to add to next samples. Breaking!" )
            rx_samples_leftovers = rx_samples_corrected[ int ( peak ): ].copy ()
            break

    if settings["log"]["verbose_1"] and real_rx : acg_vaule = pluto_rx._get_iio_attr ( 'voltage0' , 'hardwaregain' , False ) ; print ( f"{acg_vaule=}" )
    # Stop transmitting

    #corrections.estimate_cfo_drit ( rx_samples , F_S )
    #corrections.estimate_cfo_drit ( rx_samples_corrected , F_S )

    if settings["log"]["verbose_0"] : plot.plot_bpsk_symbols ( tx_bpsk_symbols , script_filename + f" {tx_bpsk_symbols.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_samples , script_filename + f" {tx_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( barker13_samples , script_filename + f" {barker13_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples , script_filename + f" {rx_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples_filtered , script_filename + f" {rx_samples_filtered.size=}" )
    if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( rx_samples_leftovers , script_filename + f" {rx_samples_leftovers.size=}" )

    ops_file.write_samples_2_csv ( settings["log"]["tx_samples"] , tx_samples )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples"] , rx_samples )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples_filtered"] , rx_samples_filtered )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples_corrected"] , rx_samples_corrected )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples_leftovers"] , rx_samples_leftovers )

if __name__ == "__main__":
    main ()
