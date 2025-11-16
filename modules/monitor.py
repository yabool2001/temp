from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import plotly.express as px

from modules import sdr

def plot_fft ( samples , f_s ) :
    psd = np.fft.fftshift ( np.abs ( np.fft.fft ( samples ) ) )
    f = np.linspace ( -f_s / 2.0 , f_s / 2.0 , len ( psd ) )
    plt
    plt.plot ( f , psd )
    plt.show ()

def plot_fft_p2 ( samples , f_s ) :
    samples = samples**2
    psd = np.fft.fftshift ( np.abs ( np.fft.fft ( samples ) ) )
    f = np.linspace ( -f_s / 2.0 , f_s / 2.0 , len ( psd ) )
    plt
    plt.plot ( f , psd )
    plt.show ()

# Funkcja do wizualizacji sygnału z offsetem (używając pandas i plotly)
def visualize_signal ( samples ) :
    analytic = signal.hilbert ( np.real ( samples ) )  # Analiza fazy z scipy
    df = pd.DataFrame ( {
        'Czas': np.arange ( len ( samples ) ) ,
        'Amplituda': np.abs ( samples ) ,
        'Faza': np.angle ( analytic )
    } )
    fig = px.line ( df , x = 'Czas' , y = [ 'Amplituda' , 'Faza' ] , title = 'Sygnał BPSK po korekcji offsetu' )
    fig.show ()  # Wyświetl interaktywny wykres (bez Orca, jeśli bez sudo)

@jit( nopython = True )
def calculate_obw_and_offset ( f , Pxx , power_percent = 0.99 ) :
    total_power = np.sum ( Pxx )
    cum_power = np.cumsum ( Pxx ) / total_power
    low_idx = np.where ( cum_power >= ( 1 - power_percent ) / 2 )[0][0]
    high_idx = np.where ( cum_power >= 1 - ( 1 - power_percent ) / 2 )[0][0]
    obw = f[ high_idx ] - f[ low_idx ]
    # Estymacja offset freq: centrum masy PSD (weighted average freq)
    freq_offset_est = np.sum ( f * Pxx ) / total_power
    return f[ low_idx ] , f[ high_idx ] , obw , freq_offset_est

def show_spectrum_occupancy_with_obw ( samples , nperseg = 1024 , power_percent = 0.99 ) :
    f_s = sdr.F_S  # 3e6 Hz
    f_c = sdr.F_C  # 3e6 Hz
    len_samples = len ( samples )
    
    if nperseg > len_samples :
        nperseg = len_samples
    noverlap = min ( nperseg // 2 , nperseg - 1 )
    
    f, Pxx = signal.welch ( samples , fs = f_s , window = 'hann' , nperseg = nperseg , 
                          noverlap = noverlap , detrend = 'constant' , scaling = 'density' )
    
    f_rf = f + f_c  # Przesunięcie o 2.9e9 Hz
    
    f_low , f_high , obw , freq_offset_est = calculate_obw_and_offset ( f_rf , Pxx , power_percent )
    print ( f"OBW ( {power_percent*100}% mocy ): {obw / 1e6:.3f} MHz" )
    print ( f"Estymowany offset freq: {freq_offset_est - f_c:.3f} Hz (od centrum baseband)")
    
    Pxx_db = 10 * np.log10 ( Pxx + 1e-12 ) # Unikanie log ( 0 )
    df = pd.DataFrame ( {'Częstotliwość [Hz]': f_rf, 'PSD [dB/Hz]': Pxx_db} )
    
    fig = px.line ( df , x = 'Częstotliwość [Hz]', y = 'PSD [dB/Hz]', title = 'PSD z OBW i offsetem (z próbek RX)' )
    fig.add_vline ( x = f_low, line_dash = "dash", line_color = "red", annotation_text = "OBW low" )
    fig.add_vline ( x = f_high, line_dash = "dash", line_color = "red", annotation_text = "OBW high" )
    fig.add_vline ( x = freq_offset_est , line_dash = "dot", line_color = "green", annotation_text = "Est. offset freq" )
    fig.update_layout ( xaxis_range = [ f_c - f_s / 2 , f_c + f_s / 2 ] , xaxis = dict ( rangeslider_visible = True ) )
    fig.show ()

def show_spectrum_occupancy ( samples , nperseg = 1024 ) :
    """
    Funkcja do wizualizacji zajętości widma (PSD) na podstawie próbek z RX PlutoSDR.
    Używa scipy.signal.welch do estymacji PSD, co jest efektywne dla sygnałów BPSK
    z dużym offsetem częstotliwości/fazy – pozwoli zobaczyć, czy sygnał jest centrowany
    wokół 0 Hz w baseband (po downconversion z rx_lo=2.9 GHz), i wykryć offsety.
    
    Parametry:
    - tsdr: Obiekt adi.Pluto z ustawionymi parametrami (sample_rate=3e6, rx_rf_bandwidth=1e6).
    - n_samples: Liczba próbek do pobrania (domyślnie rx_buffer_size=32768).
    - nperseg: Długość segmentu dla welch (trade-off: rozdzielczość vs. wariancja; mniejsza dla szybszego obliczenia).
    
    Zwraca: Interaktywny wykres PSD w dB vs. częstotliwość (w Hz, centrowana wokół rx_lo).
    """
    
    # Estymacja PSD z scipy (welch dla uśredniania, hanning window dla BPSK)
    f_s = sdr.F_S  # 3e6 Hz

    # Dynamiczne dostosowanie nperseg i noverlap
    len_samples = len ( samples )
    if nperseg > len_samples:
        nperseg = len_samples  # Automatyczna redukcja jak w scipy
    noverlap = min ( nperseg // 2 , nperseg - 1 )  # Zapewnij noverlap < nperseg
    f, Pxx = signal.welch ( samples , fs = f_s , window = 'hann' , nperseg = nperseg , noverlap = noverlap , detrend = 'constant' , scaling = 'density' )
    
    # Przesunięcie częstotliwości o rx_lo (2.9 GHz) dla wizualizacji pełnego widma RF
    f_c = sdr.F_C
    f_rf = f + f_c  # Centrowanie wokół 2.9 GHz (uwzględnia offsety)
    
    # Normalizacja do dB (dla lepszej wizualizacji zajętości widma)
    Pxx_db = 10 * np.log10 ( Pxx + 1e-12 )  # Unikanie log(0)
    
    # DataFrame do wizualizacji z pandas i plotly
    df = pd.DataFrame ( { 'Częstotliwość [Hz]': f_rf , 'PSD [dB/Hz]': Pxx_db } )
    
    # Wykres interaktywny – idealny do analizy offsetu fazy/częstotliwości w BPSK
    fig = px.line ( df , x = 'Częstotliwość [Hz]' , y = 'PSD [dB/Hz]' , title = 'Zajętość widma (PSD) sygnału RX' )
    fig.update_layout ( xaxis_title = 'Częstotliwość [Hz]' , yaxis_title = 'Moc spektralna [dB/Hz]' ,
                      xaxis_range = [ f_c - f_s / 2 , f_c + f_s / 2 ] )  # Zakres wokół lo ± fs/2
    fig.show ()
    
    # Opcjonalnie: Zintegruj z numba dla szybszego przetwarzania dużych buforów, jeśli potrzeba
    # (np. @jit na custom PSD, ale welch jest wystarczająco szybki dla N=32768)