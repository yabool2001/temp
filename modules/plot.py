import numpy as np
import pandas as pd
import plotly.express as px
from scipy import signal

from modules import sdr
from numpy.typing import NDArray

def plot_complex_waveform_v2 (samples: np.ndarray, title: str = "Samples") -> None:
    if not np.iscomplexobj ( samples ) :
        raise ValueError ( "Wejściowy sygnał musi być zespolony (np.ndarray typu complex)." )

    # Tworzenie DataFrame do wykresu
    df = pd.DataFrame ( { "index": np.arange ( len ( samples ) ) , "real": samples.real , "imag": samples.imag } )
    fig = px.line(df, x="index", y=["real", "imag"], labels={"variable": f"{title} ( {samples.size=} )"} )

    # Modyfikacja nazw, stylu i włączenie legendy
    fig.update_traces(selector=dict(name="real"), name="I (real)" )
    fig.update_traces(selector=dict(name="imag"), name="Q (imag)", line=dict(dash="dash") )

    # Ustawienia osi    
    fig.update_layout(
        xaxis_title="Numer próbki",
        yaxis_title="Amplituda",
        xaxis = dict ( rangeslider_visible = True ) ,
        #legend = dict ( orientation="h" , yanchor="bottom" , xanchor="center" , x = 0.01 , y = 0.99 ) ,
        legend = dict ( orientation="h" , yanchor = "bottom" , y = 1.02 , xanchor = "center" , x = 0.5 ) ,
        height = 400
    )
    fig.show()

def plot_complex_waveform(signal_complex: np.ndarray, title: str = "Sygnał BPSK po modulacji i filtracji") -> None:
    """
    Rysuje wykres rzeczywistej i urojonej części sygnału zespolonego.
    
    Parametry:
    ----------
    signal_complex : np.ndarray
        Sygnał zespolony po modulacji BPSK i filtracji (np. RRC).
    title : str
        Tytuł wykresu (opcjonalnie).

    Zwraca:
    -------
    None
    """
    # Upewnij się, że wejście to tablica zespolona
    if not np.iscomplexobj(signal_complex):
        raise ValueError("Wejściowy sygnał musi być zespolony (np.ndarray typu complex).")

    # Tworzenie DataFrame do wykresu
    df = pd.DataFrame({
        "index": np.arange(len(signal_complex)),
        "real": signal_complex.real,
        "imag": signal_complex.imag
    })

    # Rysowanie wykresu
    fig = px.line(df, x="index", y="real", title=title)
    fig.add_scatter(x=df["index"], y=df["imag"], mode="lines", name="Q (imag)", line=dict(dash="dash"))
    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( x = 0.01 , y = 0.99 ) ,
        height = 500
    )
    fig.show()


def complex_waveform ( signal_complex: np.ndarray, title: str = "Sygnał zespolony", marker_squares: bool = False) -> None:
    """
    Rysuje wykres rzeczywistej i urojonej części sygnału zespolonego.

    Dodatkowy parametr `marker_squares`: jeśli True, próbki zostaną oznaczone małymi kwadratami.

    Parametry:
    - signal_complex: np.ndarray (complex)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na próbkach
    """
    if not np.iscomplexobj(signal_complex):
        raise ValueError("Wejściowy sygnał musi być zespolony (np.ndarray typu complex).")

    df = pd.DataFrame({
        "index": np.arange(len(signal_complex)),
        "real": signal_complex.real,
        "imag": signal_complex.imag
    })

    # Wybór trybu i markerów
    if marker_squares:
        mode_real = 'lines+markers'
        mode_imag = 'lines+markers'
        # hollow square markers, size increased by 1 (was 4 -> now 5)
        marker_real_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='blue', width=1))
        marker_imag_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='orange', width=1))
    else:
        mode_real = 'lines'
        mode_imag = 'lines'
        marker_real_cfg = None
        marker_imag_cfg = None

    fig = px.line(df, x="index", y="real", title=title)
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter(x=df["index"], y=df["real"], mode=mode_real, name="I (real)", line=dict(color='blue'), marker=marker_real_cfg)
    fig.add_scatter(x=df["index"], y=df["imag"], mode=mode_imag, name="Q (imag)", line=dict(color='orange', dash='dash'), marker=marker_imag_cfg)

    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( x = 0.01 , y = 0.99 ) ,
        height = 500
    )
    fig.show()

def plot_bpsk_symbols(symbols: np.ndarray, title: str = "Symbole BPSK", filename: str = "–") -> None:
    """
    Rysuje wykres symboli BPSK w postaci punktów połączonych przerywaną linią.
    
    Parametry:
    ----------
    symbols : np.ndarray
        Tablica symboli BPSK (+1 / -1).
    title : str
        Tytuł wykresu.
    filename : str
        Nazwa pliku źródłowego (do wyświetlenia w tytule, opcjonalna dekoracja).

    Zwraca:
    -------
    None
    """
    if not isinstance(symbols, np.ndarray):
        raise TypeError("Argument 'symbols' musi być typu numpy.ndarray.")

    # Przygotowanie danych do wykresu
    df = pd.DataFrame({
        "symbol_index": np.arange(len(symbols)),
        "symbol": symbols
    })

    # Wykres punktowy
    fig = px.scatter(
        df,
        x="symbol_index",
        y="symbol",
        title=f"{title} z pliku {filename}",
        labels={"symbol": "Wartość symbolu", "symbol_index": "Indeks symbolu"}
    )

    # Dodanie przerywanej linii łączącej punkty
    fig.add_scatter(
        x=df["symbol_index"],
        y=df["symbol"],
        mode='lines+markers',
        name='Symbole BPSK',
        line=dict(color='gray', width=1, dash='dot')
    )

    # Konfiguracja osi i wyglądu
    fig.update_layout(
        height=500,
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99)
    )

    # Ustawienie skali osi Y
    fig.update_yaxes(range=[-1.5, 1.5])

    # Wyświetlenie wykresu
    fig.show()

def plot_symbols ( symbols : NDArray[ np.complex128 ] , title : str = "Symbole BPSK" ) -> None :
    """
    Rysuje wykres symboli BPSK (np. z ADALM-Pluto) w postaci punktów połączonych przerywaną linią.

    Parametry:
    ----------
    symbols : NDArray[ np.complex128 ] Tablica symboli BPSK typu np.complex128
    title : str Tytuł wykresu.
    Zwraca: None
    """
    if not isinstance ( symbols , np.ndarray ):
        raise TypeError ( "Argument 'symbols' musi być typu NDArray[np.complex128]." )

    # Obsługa symboli zespolonych – bierzemy część rzeczywistą
    symbols_real = symbols.real if np.iscomplexobj ( symbols ) else symbols
    # Przygotowanie danych do wykresu
    df = pd.DataFrame ( { "symbol_index" : np.arange ( len ( symbols_real ) ) , "symbol" : symbols_real } )
    # Wykres punktowy
    fig = px.scatter ( df , x = "symbol_index" , y = "symbol" , title = f"{ title }" , labels = { "symbol" : "Wartość symbolu" , "symbol_index" : "Indeks symbolu" } )
    # Dodanie przerywanej linii łączącej punkty
    fig.add_scatter ( x = df[ "symbol_index" ] , y = df[ "symbol" ] , mode = 'lines+markers' , name = 'Symbole' , line = dict ( color = 'gray' , width = 1 , dash = 'dot' ) )
    # Konfiguracja osi i wyglądu
    fig.update_layout ( height = 500 , xaxis = dict ( rangeslider_visible = True ) , legend = dict ( x = 0.01 , y = 0.99 ) )
    # Oś Y dopasowana dynamicznie, ale możesz wymusić np. range=[-1.5, 1.5] jeśli chcesz sztywną skalę
    fig.show ()

def plot_bpsk_symbols_v2(symbols: np.ndarray,
                      title: str = "Symbole BPSK",
                      filename: str = "–") -> None:
    """
    Rysuje wykres symboli BPSK (np. z ADALM-Pluto) w postaci punktów połączonych przerywaną linią.

    Parametry:
    ----------
    symbols : np.ndarray
        Tablica symboli BPSK, może być typu float, int, complex (np. complex128).
    title : str
        Tytuł wykresu.
    filename : str
        Nazwa pliku źródłowego (opcjonalna dekoracja w tytule).

    Zwraca:
    -------
    None
    """
    if not isinstance(symbols, np.ndarray):
        raise TypeError("Argument 'symbols' musi być typu numpy.ndarray.")

    # Obsługa symboli zespolonych – bierzemy część rzeczywistą
    symbols_real = symbols.real if np.iscomplexobj(symbols) else symbols

    # Przygotowanie danych do wykresu
    df = pd.DataFrame({
        "symbol_index": np.arange(len(symbols_real)),
        "symbol": symbols_real
    })

    # Wykres punktowy
    fig = px.scatter(
        df,
        x="symbol_index",
        y="symbol",
        title=f"{title} z pliku {filename}",
        labels={"symbol": "Wartość symbolu", "symbol_index": "Indeks symbolu"}
    )

    # Dodanie przerywanej linii łączącej punkty
    fig.add_scatter(
        x=df["symbol_index"],
        y=df["symbol"],
        mode='lines+markers',
        name='Symbole BPSK',
        line=dict(color='gray', width=1, dash='dot')
    )

    # Konfiguracja osi i wyglądu
    fig.update_layout(
        height=500,
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99)
    )

    # Oś Y dopasowana dynamicznie, ale możesz wymusić np. range=[-1.5, 1.5] jeśli chcesz sztywną skalę
    fig.show()

def spectrum_occupancy ( samples , nperseg = 1024 , title: str = "Spectrum occupancy (PSD)" ) -> None :
    """
    Funkcja do wizualizacji zajętości widma (PSD) na podstawie próbek.
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
    fig = px.line ( df , x = 'Częstotliwość [Hz]' , y = 'PSD [dB/Hz]' , title = title )
    fig.update_layout ( xaxis_title = 'Częstotliwość [Hz]' , yaxis_title = 'Moc spektralna [dB/Hz]' ,
                      xaxis_range = [ f_c - f_s / 2 , f_c + f_s / 2 ] )  # Zakres wokół lo ± fs/2
    fig.show ()
    
    # Opcjonalnie: Zintegruj z numba dla szybszego przetwarzania dużych buforów, jeśli potrzeba
    # (np. @jit na custom PSD, ale welch jest wystarczająco szybki dla N=32768)