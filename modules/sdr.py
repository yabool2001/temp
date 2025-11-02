import adi
import iio
import json
import plotly.express as px
#import matplotlib.pyplot as plt
#from modules import filters
import numpy as np
import os
import time 

with open ( "settings.json" , "r" ) as settings_file :
    settings = json.load ( settings_file )
    pluto = settings[ "ADALM-Pluto" ]

TX_SN = pluto[ "URI" ][ "SN_TX" ]
RX_SN = pluto[ "URI" ][ "SN_RX" ]
F_C = int ( pluto[ "F_C" ] )    # Carrier frequency [Hz]
BW  = int ( pluto[ "BW" ] )     # BandWidth [Hz]
#F_S = 521100     # Sampling frequency [Hz] >= 521e3 && <
F_S = int ( BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100 ) # Sampling frequency [Hz]
TX_GAIN = float ( pluto[ "TX_GAIN" ] )
RX_GAIN = int ( pluto[ "RX_GAIN" ] )
GAIN_CONTROL = pluto[ "GAIN_CONTROL" ]
SAMPLES_BUFFER_SIZE = int ( pluto[ "SAMPLES_BUFFER_SIZE" ] )

def init_pluto_v3 ( sn ) :
    uri = get_uri ( sn )
    sdr = adi.Pluto ( uri )
    sdr.tx_lo = int ( F_C )
    sdr.rx_lo = int ( F_C )
    sdr.sample_rate = int ( F_S )
    sdr.rx_rf_bandwidth = int ( BW )
    sdr.rx_buffer_size = int ( SAMPLES_BUFFER_SIZE )
    sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
    sdr.gain_control_mode_chan0 = GAIN_CONTROL
    sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
    sdr.rx_output_type = "SI"
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    time.sleep ( 0.2 ) #delay after setting device parameters
    if settings["log"]["verbose_2"] : print ( f"{F_C=} {BW=} {F_S=}" )
    return sdr


def init_pluto_v2 ( uri , f_c , f_s , bw , tx_gain) :
    sdr = adi.Pluto ( uri )
    sdr.tx_lo = int ( f_c )
    sdr.rx_lo = int ( f_c )
    sdr.sample_rate = int ( f_s )
    sdr.rx_rf_bandwidth = int ( bw )
    sdr.rx_buffer_size = int ( SAMPLES_BUFFER_SIZE )
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
    sdr.rx_buffer_size = int ( SAMPLES_BUFFER_SIZE )
    sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
    sdr.gain_control_mode_chan0 = GAIN_CONTROL
    sdr.rx_hardwaregain_chan0 = float ( RX_GAIN )
    sdr.rx_output_type = "SI"
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    time.sleep ( 0.2 ) #delay after setting device parameters
    return sdr

def tx_once ( samples , sdr ) :
    sdr.tx_destroy_buffer ()
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    sdr.tx_cyclic_buffer = False
    sdr.tx ( samples )

def tx_cyclic ( samples , sdr ) :
    sdr.tx_destroy_buffer () # Dodałem to w wersji ok. v0.1.1 ale nie wiem czy to dobrze
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    sdr.tx_cyclic_buffer = True
    sdr.tx ( samples )

def stop_tx_cyclic ( sdr ) :
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    print ( f"{sdr.tx_cyclic_buffer=}" )

def rx_samples ( sdr ) :
    return sdr.rx ()

def analyze_rx_signal ( samples ) :
    # Real vs Imag plot
    real = samples.real[:500]
    imag = samples.imag[:500]
    fig1 = px.line(y=[real, imag], title="Real vs Imag")
    fig1.update_traces(name='Real', selector=dict(name='0'))
    fig1.update_traces(name='Imag', selector=dict(name='1'))
    fig1.update_layout(showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True)
    fig1.show()

    # Constellation plot
    fig2 = px.scatter(x=samples.real, y=samples.imag, opacity=0.3, title="Constellation")
    fig2.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    fig2.show()

    # Histogram amplitudy
    fig3 = px.histogram(x=np.abs(samples), nbins=100, title="Histogram amplitudy")
    fig3.show()
"""
def analyze_rx_signal_old ( samples ) :
    plt.plot(samples.real[:500])
    plt.plot(samples.imag[:500])
    plt.title("Real vs Imag")
    plt.grid()
    plt.scatter(samples.real, samples.imag, alpha=0.3)
    plt.axis('equal')
    plt.title("Constellation")
    plt.hist(np.abs(samples), bins=100)
    plt.title("Histogram amplitudy")
"""
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

def validate_samples ( samples: np.ndarray , buffer_size ) :

    validation = True

    # Walidacja: typ danych
    if not isinstance(samples, np.ndarray):
        raise ValueError("❌ tx_samples is not a numpy array")
        validation = False

    if samples.dtype != np.complex128:
        raise ValueError(f"❌ tx_samples must be np.complex128, but got {samples.dtype}")
        validation = False

    # Walidacja: wymiar
    if samples.ndim != 1:
        raise ValueError("❌ tx_samples must be a 1D array")
        validation = False

    # Walidacja: zawartość
    if np.isnan(samples).any():
        raise ValueError("❌ tx_samples contains NaN values")
        validation = False

    if np.isinf(samples).any():
        raise ValueError("❌ tx_samples contains Inf values")
        validation = False
    
    if samples.size > buffer_size :
        raise ValueError("❌ tx_samples size is larger than sdr buffer size")
        validation = False

    return validation

