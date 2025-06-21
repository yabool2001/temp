import numpy as np
import os
import pandas as pd
import plotly.express as px
from scipy.signal import lfilter, correlate , upfirdn
from scipy.signal.windows import hamming
from numba import njit  # Dodane dla przyspieszenia obliczeń

# Inicjalizacja plików CSV
csv_filename_tx_waveform = "complex_tx_waveform.csv"
csv_filename_rx_waveform = "complex_rx_waveform.csv"

script_filename = os.path.basename ( __file__ )

SPS = 4                 # próbek na symbol
RRC_BETA = 0.35         # roll-off factor
RRC_SPAN = 11           # długość filtru RRC w symbolach

FAKE_BITS = np.array ( [ 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ] , dtype = np.float64 )
BARKER13_BITS = np.array ( [ 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 ] , dtype = np.float64 )
PADDING_BITS = np.array ( [ 0 , 0 , 0 ] , dtype = np.float64 )
PAYLOAD_BITS = np.array ( [ 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 ] , dtype = np.float64 )


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
    fig.update_layout(
        xaxis_title="Numer próbki",
        yaxis_title="Amplituda",
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99),
        height=500
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

@njit ( cache = True , fastmath = True )  # Kompilacja Just-In-Time z optymalizacjami
def rrc_filter_v4 ( beta , sps , span ) :
    """
    DeepSeek -V3 R1 (Zoptymalizowana)
    Filtruje dane wejściowe za pomocą filtra RRC.
    """
    N = span * sps
    t = np.arange ( -N / 2 , N / 2 + 1 , dtype = np.float64 ) / sps

    # Obsługa beta = 0 (filtr sinc)
    if beta == 0 :
        h = np.sinc ( t )
        h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja
        return h
    
    h = np.zeros_like ( t )

    # Stałe pre-kalkulowane dla wydajności
    beta_pi = np.pi * beta
    inv_4beta = 1 / ( 4 * beta )
    sqrt2 = np.sqrt ( 2 )
    special_val = ( beta / sqrt2 ) * \
                ( ( 1 + 2 / np.pi ) * np.sin ( np.pi / ( 4 * beta ) ) + 
                ( 1 - 2 / np.pi ) * np.cos ( np.pi / ( 4 * beta ) ) )

    # Obliczenia z maskami (zoptymalizowane)
    for i in range ( len ( t ) ) :
        ti = t[i]
        if ti == 0.0 :
            h[i] = 1.0 - beta + ( 4 * beta / np.pi )
        elif np.abs ( np.abs ( ti ) - inv_4beta ) < 1e-10 :  # Tolerancja numeryczna
            h[i] = special_val
        else :
            numerator = np.sin ( np.pi * ti * ( 1 - beta ) ) + \
                      4 * beta * ti * np.cos ( np.pi * ti * ( 1 + beta ) )
            denominator = np.pi * ti * ( 1 - ( 4 * beta * ti ) ** 2 )
            h[i] = numerator / denominator

    # Bezpieczna obsługa NaN/Inf i normalizacja
    h[np.isnan ( h ) | np.isinf ( h )] = 0
    h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja

    return h

def apply_tx_rrc_filter ( symbols: np.ndarray , sps: int = 4 , beta: float = 0.35, span: int = 11 , upsample: bool = True , ) -> np.ndarray :
    """
    Stosuje filtr Root Raised Cosine (RRC) z opcjonalnym upsamplingiem.
    Zawsze zwraca sygnał zespolony (complex128).

    Parametry:
        symbols: Sygnał wejściowy (real lub complex).
        beta: Współczynnik roll-off (0.0-1.0).
        sps: Próbek na symbol (samples per symbol).
        span: Długość filtra w symbolach.
        upsample: Czy wykonać upsampling (True) czy tylko filtrować (False).

    Zwraca:
        Przefiltrowany sygnał zespolony (complex128).
    """
    rrc_taps = rrc_filter_v4 ( beta , sps , span )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , sps )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )  # Wymuszenie complex128

def main():
    tx_bits = np.concatenate ( [ BARKER13_BITS , PADDING_BITS , PAYLOAD_BITS , BARKER13_BITS , PADDING_BITS , PAYLOAD_BITS ] )
    print ( f"{tx_bits=}" )
    barker13_symbols = 2 * BARKER13_BITS - 1
    print ( f"{barker13_symbols=}" )
    barker13_samples = apply_tx_rrc_filter ( barker13_symbols , SPS , RRC_BETA , RRC_SPAN , True )
    plot_complex_waveform ( barker13_samples , script_filename + " barker13__samples" )
    tx_bpsk_symbols = 2 * tx_bits - 1
    print ( f"{tx_bpsk_symbols=}" )
    #plot_bpsk_symbols ( tx_bpsk_symbols )
    tx_samples = apply_tx_rrc_filter ( tx_bpsk_symbols , SPS , RRC_BETA , RRC_SPAN , True )
    plot_complex_waveform ( tx_samples , script_filename + " tx_samples" )

    # Receive samples
    rx_samples = tx_samples

    # Filtracja RRC sygnału RX
    rx_filtered_samples = apply_tx_rrc_filter ( rx_samples , SPS , RRC_BETA , RRC_SPAN , upsample = False )
    plot_complex_waveform ( rx_filtered_samples , script_filename + " rx_filtered_samples" )
    corr = np.correlate ( rx_filtered_samples , barker13_samples , mode = 'full' )
    peak_index = np.argmax ( np.abs ( corr ) )
    print ( f"{peak_index=}" )
    # Przesunięcie związane z pełną korelacją
    timing_offset = peak_index - len ( barker13_samples ) + 1
    print ( f"{timing_offset=}" )
    aligned_rx_samples = rx_filtered_samples[ timing_offset: ]
    plot_complex_waveform ( aligned_rx_samples , script_filename + " aligned_rx_samples" )
    # Odczytywanie symboli co SPS próbek, zaczynając od środka symbolu
    symbols_rx = aligned_rx_samples [ RRC_SPAN * SPS // 2::SPS]
    print ( f"{symbols_rx=}" )
    #plot_bpsk_symbols ( symbols_rx , "symbols_rx    " )
    bits_rx = ( symbols_rx.real > 0 ).astype ( int )
    print ( f"{bits_rx=}" )

if __name__ == "__main__":
    main ()
