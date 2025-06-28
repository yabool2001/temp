import adi
import iio
import matplotlib.pyplot as plt
from modules import filters
import numpy as np
import time 

TX_GAIN = -10
RX_GAIN = 70
GAIN_CONTROL = "slow_attack"
#GAIN_CONTROL = "fast_attack"
#GAIN_CONTROL = "manual"
NUM_SAMPLES = 10000
#NUM_SAMPLES = 32768

def init_pluto_v2 ( uri , f_c , f_s , bw , tx_gain) :
    sdr = adi.Pluto ( uri )
    sdr.tx_lo = int ( f_c )
    sdr.rx_lo = int ( f_c )
    sdr.sample_rate = int ( f_s )
    sdr.rx_rf_bandwidth = int ( bw )
    sdr.rx_buffer_size = int ( NUM_SAMPLES )
    sdr.tx_hardwaregain_chan0 = float ( tx_gain )
    sdr.gain_control_mode_chan0 = GAIN_CONTROL
    sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
    sdr.rx_output_type = "SI"
    #sdr.tx_destroy_buffer ()
    #sdr.tx_cyclic_buffer = False
    time.sleep ( 0.2 ) #delay after setting device parameters
    return sdr

def init_pluto ( uri , f_c , f_s , bw ) :
    sdr = adi.Pluto ( uri )
    sdr.tx_lo = int ( f_c )
    sdr.rx_lo = int ( f_c )
    sdr.sample_rate = int ( f_s )
    sdr.rx_rf_bandwidth = int ( bw )
    sdr.rx_buffer_size = int ( NUM_SAMPLES )
    sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
    sdr.gain_control_mode_chan0 = GAIN_CONTROL
    sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
    sdr.rx_output_type = "SI"
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    time.sleep ( 0.2 ) #delay after setting device parameters
    return sdr

def tx_once ( samples , sdr ) :
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    sdr.tx ( samples )

def tx_cyclic ( samples , sdr ) :
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    sdr.tx_cyclic_buffer = True
    sdr.tx ( samples )

def stop_tx_cyclic ( sdr ) :
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    #print ( f"{sdr.tx_cyclic_buffer=}" )

def rx_samples ( sdr ) :
    return sdr.rx ()

def rx_samples_filtered ( sdr , sps = 8 , beta = 0.35 , span = 11 ) :
    return filters.apply_rrc_filter ( rx_samples ( sdr ) , sps , beta , span )

def analyze_rx_signal ( samples ) :
    plt.plot(samples.real[:500])
    plt.plot(samples.imag[:500])
    plt.title("Real vs Imag")
    plt.grid()
    plt.scatter(samples.real, samples.imag, alpha=0.3)
    plt.axis('equal')
    plt.title("Constellation")
    plt.hist(np.abs(samples), bins=100)
    plt.title("Histogram amplitudy")

def get_uri ( serial: str , type_preference: str = "usb" ) -> str | None :
    """
    Zwraca URI kontekstu IIO dla danego numeru seryjnego.
    
    Arguments:
    - serial (str): numer seryjny urządzenia (pełny).
    - type_preference (str): "usb" lub "ip". Jeśli "ip", preferuje ip: ale wraca do usb: jeśli ip nie znaleziono.

    Returns:
    - str: URI w formacie usb:x.y.z lub ip:adres
    - None: jeśli nie znaleziono pasującego urządzenia
    """
    contexts = iio.scan_contexts()

    ip_match = None
    usb_match = None

    for uri, description in contexts.items():
        if serial in description:
            if uri.startswith("ip:") and type_preference == "ip":
                ip_match = uri
            elif uri.startswith("usb:"):
                usb_match = uri

    if type_preference == "ip":
        return ip_match or usb_match
    elif type_preference == "usb":
        return usb_match

    return None
