'''
Issue #8: Tworzenie dokumentacji dla różnych parametrów rrc_filter

Najlepiej działa z bpsk-real-tx_v3-curses.py
2025.10.20 Zmiany wprowadzone w celu wdrożenia sdr.init_pluto_v3
2025.10.29 Wprowadzić obsługę zmiennej App setting: cuda

Frame structure: [ preamble_bits , header_bits , payload_bits , crc32_bits ]
preamble_bit    [ 6 , 80 ]          2 bytes of fixed value preamble: 13 bits of BARKER 13 + 3 bits of padding
header_bits     [ X ]               1 byte of payload length = header value + 1
payload_bits    [ X , ... ]         variable length payload - max 256
crc32_bits      [ X , X , X , X ]   4 bytes of payload CRC32 
'''

import json
import numpy as np
import os
import time as t

from modules import filters , sdr , ops_packet , ops_file , modulation , monitor , corrections , plot
#from modules.rrc import rrc_filter
#from modules.clock_sync import polyphase_clock_sync

script_filename = os.path.basename ( __file__ )
# Wczytaj plik JSON z konfiguracją
with open ( "settings.json" , "r" ) as settings_file :
    settings = json.load ( settings_file )

### App settings ###
real_rx = True # Pobieranie żywych danych z Pluto 
cuda = True
#real_rx = False # Ładowanie danych zapisanych w pliku:

#rx_saved_filename = "logs/rx_samples_10k.csv"
#rx_saved_filename = "logs/rx_samples_32768.csv"
#rx_saved_filename = "logs/rx_samples_1255-barely_payload.csv"
rx_saved_filename = "logs/rx_samples_1240-no_payload.csv"
#rx_saved_filename = "logs/rx_samples_987-no_crc32.csv"
#rx_saved_filename = "logs/rx_samples_702-no_preamble.csv"
#rx_saved_filename = "logs/rx_samples_1245-no_barker.csv"

PAYLOAD = [ 0x0F , 0x0F , 0x0F , 0x0F ]  # aktualnie statyczny 4-bajtowy ładunek używany do testów

# ------------------------ KONFIGURACJA SDR ------------------------
def main() :
    packet_bits = ops_packet.create_packet_bits ( PAYLOAD )
    tx_bpsk_symbols = modulation.create_bpsk_symbols ( packet_bits )
    tx_samples = filters.apply_tx_rrc_filter_v0_1_3 ( tx_bpsk_symbols , True )
    if real_rx :
        pluto_rx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_RX"] )
        for i in range ( 0 , 100 ) : # Clear buffer just to be safe
            raw_data = sdr.rx_samples ( pluto_rx )
        if settings["log"]["verbose_0"] : monitor.plot_fft_p2 ( raw_data , sdr.F_S )
    else :
        rx_samples = ops_file.open_csv_and_load_np_complex128 ( rx_saved_filename ) # Nie powinno w pętli.
    # Receive and process samples
    barker13_samples = modulation.get_barker13_bpsk_samples_v0_1_3 ( True )
    print ( "Start Rx!" ) 
    while True :
        if real_rx :
            rx_samples = sdr.rx_samples ( pluto_rx )
        if ops_packet.is_sync_seq ( rx_samples , barker13_samples ) :
            print ( "Yes!" )
            monitor.show_spectrum_occupancy_with_obw ( rx_samples , nperseg = 1024 )
            #sdr.stop_tx_cyclic ( pluto_tx )
            start = t.perf_counter ()
            rx_samples_filtered = filters.apply_rrc_rx_filter_v0_1_3 ( rx_samples , False )
            end = t.perf_counter ()
            print ( f"apply_rrc_rx_filter perf: {end - start:.6f} sekundy" )
            start = t.perf_counter ()
            rx_samples_corrected = corrections.full_compensation_v0_1_5 ( rx_samples_filtered , modulation.get_barker13_bpsk_samples_v0_1_3 ( True ) )
            end = t.perf_counter ()
            print ( f"full_compensation perf: {end - start:.6f} sekundy" )
            start = t.perf_counter ()
            rx_samples_corrected = modulation.zero_quadrature ( rx_samples_corrected )
            end = t.perf_counter ()
            print ( f"zero_quadrature perf: {end - start:.6f} sekundy" )
            if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( rx_samples_corrected , script_filename + f" {rx_samples_corrected.size=}" )
            if settings["log"]["verbose_1"] : print ( f"{rx_samples_corrected.size=} ")
            start = t.perf_counter ()
            corr = modulation.fast_normalized_cross_correlation ( rx_samples_corrected , barker13_samples )
            end = t.perf_counter ()
            print ( f"corr perf: {end - start:.6f} sekundy" )
            threshold = np.mean ( corr ) + 3 * np.std ( corr )
            detected_peaks = np.where(corr >= threshold)[0]
            peaks = modulation.group_peaks_by_distance ( detected_peaks , corr , min_distance = 2 )
            if settings["log"]["verbose_1"] : print ( f"\n{peaks.size=} | {rx_samples_corrected.size=}")
            for peak in peaks :
                if ops_packet.is_preamble_v0_1_3 ( rx_samples_corrected[ peak: ] ) :
                    # Upewnij się, że payload_bits jest zawsze zainicjowane, bo w przypadku wyjątku poniżej zmienna mogłaby nie istnieć i byłby błąd
                    payload_bits = None
                    try: # Wstawiłem to 24.06.2025, żeby rozkminić błąd TypeError: cannot unpack non-iterable NoneType object
                        payload_bits = ops_packet.get_payload_bytes_v0_1_3 ( rx_samples_corrected[ peak: ] )
                    except Exception as e:
                        if settings["log"]["verbose_2"]: print ( f"get_payload_bytes error: {e}" )
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

    if settings["log"]["verbose_0"] : plot.plot_bpsk_symbols ( tx_bpsk_symbols , script_filename + f" {tx_bpsk_symbols.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( tx_samples , script_filename + f" {tx_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( barker13_samples , script_filename + f" {barker13_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples , script_filename + f" {rx_samples.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples_filtered , script_filename + f" {rx_samples_filtered.size=}" )
    if settings["log"]["verbose_1"] : plot.plot_complex_waveform ( rx_samples_corrected , script_filename + f" {rx_samples_corrected.size=}" )
    if settings["log"]["verbose_2"] : plot.plot_complex_waveform ( rx_samples_leftovers , script_filename + f" {rx_samples_leftovers.size=}" )

    ops_file.write_samples_2_csv ( settings["log"]["tx_samples"] , tx_samples )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples"] , rx_samples )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples_filtered"] , rx_samples_filtered )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples_corrected"] , rx_samples_corrected )
    ops_file.write_samples_2_csv ( settings["log"]["rx_samples_leftovers"] , rx_samples_leftovers )

if __name__ == "__main__":
    main ()
