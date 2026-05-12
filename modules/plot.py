import numpy as np
import pandas as pd
import plotly.express as px
import torch
from scipy import signal
from typing import Optional, TYPE_CHECKING

from modules import modulation, sdr
from numpy.typing import NDArray

if TYPE_CHECKING:
    from modules import packet

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

def complex_symbols ( complex_symbols : NDArray[ np.complex128 ] , title : str = "Symbole zespolone" ) -> None:
    #fig = px.scatter ( complex_symbols , x = complex_symbols.real , y = complex_symbols.imag )
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex) / np.sqrt(2)
    labels = np.array(['00', '01', '11', '10'])[bits]
    fig = px.scatter(
        x=complex_symbols,                               # bezpośrednio liczby zespolone!
        color=labels,
        symbol=labels,
        color_discrete_sequence=['#FFD700', '#FFEA00', '#C0FF00', '#BFFF00'],  # złoto i limonka
        symbol_sequence=['circle', 'square', 'diamond', 'x'],
        opacity=0.78,
        size_max=7,
        title=f"<b>Konstelacja QPSK</b><br>"
            f"E<sub>b</sub>/N<sub>0</sub> = {EbN0_dB} dB  •  {N:,} symboli",
    )

    # Idealne punkty – duże, złote z czarną obwódką
    fig.add_scatter(
        x=constellation,
        mode="markers+text",
        marker=dict(size=28, color="#FFD700", line=dict(width=4, color="black")),
        text=['00', '01', '11', '10'],
        textposition="middle center",
        textfont=dict(color="black", size=14, family="Arial Black"),
        name="Idealne symbole"
    )
    fig.update_xaxes(
    range=[-1.7, 1.7],
    zeroline=True, zerolinewidth=3, zerolinecolor="#444",
    showgrid=True, gridwidth=1, gridcolor="#222",
    ticks="outside", tickcolor="#555", ticklen=8,
    title="", linewidth=2, linecolor="#555"
)

    fig.update_yaxes(
        range=[-1.7, 1.7],
        zeroline=True, zerolinewidth=3, zerolinecolor="#444",
        showgrid=True, gridwidth=1, gridcolor="#222",
        scaleanchor="x", scaleratio=1,
        ticks="outside", tickcolor="#555", ticklen=8,
        title="", linewidth=2, linecolor="#555"
    )

    fig.update_layout(
        width=900, height=900,
        plot_bgcolor="#0E1117",      # głęboka czerń tła wykresu
        paper_bgcolor="#000000",     # czerń papieru
        font=dict(color="#DDDDDD", size=14, family="Arial"),
        title=dict(x=0.5, xanchor="center", y=0.95),
        legend=dict(
            title="Bity (Gray)",
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#444",
            borderwidth=2,
            font=dict(color="#FFD700")
        )
    )

    # Subtelny złoty okrąg jednostkowy
    fig.add_shape(
        type="circle", xref="x", yref="y",
        x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color="#FFD700", width=2, dash="solid"),
        opacity=0.4
    )
    #fig.update_xaxes ( range = [ -1.8 , 1.8 ] , zeroline = True , zerolinewidth = 2 , zerolinecolor = "#333" )
    #fig.update_yaxes ( range = [ -1.8 , 1.8 ] , zeroline = True , zerolinewidth = 2 , zerolinecolor="#333" , scaleanchor = "x" , scaleratio = 1 )
    #fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=1, color='black')))
    #fig.add_shape ( type = "circle" , x0 = -1 , y0 = -1 , x1 = 1 , y1 = 1 , line_color = "red" , line_dash = "dash" )  # okrąg jednostkowy
    fig.show ()


def complex_waveform ( signal_complex: NDArray[ np.complex128 ] , title: str = "Sygnał zespolony" , marker_squares : bool = False ) -> None:
    """
    Rysuje wykres rzeczywistej i urojonej części sygnału zespolonego.

    Dodatkowy parametr `marker_squares`: jeśli True, próbki zostaną oznaczone małymi kwadratami.

    Parametry:
    - signal_complex: NDArray[ np.complex128 ] (complex)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na próbkach
    """
    if not np.iscomplexobj ( signal_complex ) :
        raise ValueError ( "Wejściowy sygnał musi być zespolony NDArray[ np.complex128 ]" )

    df = pd.DataFrame ( { "index" : np.arange ( len ( signal_complex ) ) , "real" : signal_complex.real , "imag" : signal_complex.imag } )

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

    fig = px.line ( df , x = "index" , y = "real" , title = f"{ title }" )
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter(x=df["index"], y=df["real"], mode=mode_real, name="I (real)", line=dict(color='blue'), marker=marker_real_cfg)
    fig.add_scatter(x=df["index"], y=df["imag"], mode=mode_imag, name="Q (imag)", line=dict(color='green', dash='dash'), marker=marker_imag_cfg)

    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( x = 0.01 , y = 0.99 ) ,
        height = 500
    )
    fig.show()

def real_waveform ( signal_real : NDArray[ np.float64 ] , title : str = "Sygnał rzeczywisty" , marker_squares : bool = False ) -> None :
    """
    Rysuje wykres wartości rzeczywistych sygnału.

    Dodatkowy parametr `marker_squares`: jeśli True, próbki zostaną oznaczone małymi kwadratami.

    Parametry:
    - signal_real: NDArray[ np.float64 ] (real)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na próbkach
    """
    if np.iscomplexobj ( signal_real ) :
        raise ValueError ( "Wejściowy sygnał musi być rzeczywisty NDArray[ np.float64 ]" )

    df = pd.DataFrame ( { "index" : np.arange ( len ( signal_real ) ) , "value" : signal_real } )

    # Wybór trybu i markerów
    if marker_squares:
        mode = 'lines+markers'
        marker_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='blue', width=1))
    else:
        mode = 'lines'
        marker_cfg = None

    fig = px.line ( df , x = "index" , y = "value" , title = f"{ title }" )
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter(x=df["index"], y=df["value"], mode=mode, name="Wartość", line=dict(color='blue'), marker=marker_cfg)

    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( x = 0.01 , y = 0.99 ) ,
        height = 500
    )
    fig.show()

def real_waveform_v0_1_6(signal_real: NDArray[np.float64], title: str = "Sygnał rzeczywisty", marker_squares: bool = False, marker_peaks: Optional[NDArray[np.int_]] = None) -> None:
    """
    Rozszerzona wersja funkcji real_waveform z dodatkowym parametrem marker_peaks.
    Jeśli marker_peaks zostanie przekazany (np.ndarray z indeksami), peaks zostaną zaznaczone trójkątami na wykresie.

    Parametry:
    - signal_real: NDArray[np.float64] (rzeczywisty)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na wszystkich próbkach
    - marker_peaks: Optional[NDArray[np.int_]] — indeksy próbek, gdzie zaznaczyć trójkąty (rozmiar taki sam jak marker_squares)
    """
    if np.iscomplexobj(signal_real):
        raise ValueError("Wejściowy sygnał musi być rzeczywisty NDArray[np.float64]")

    df = pd.DataFrame({"index": np.arange(len(signal_real)), "value": signal_real})

    if marker_squares:
        mode = 'lines+markers'
        marker_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='blue', width=1))
    else:
        mode = 'lines'
        marker_cfg = None

    fig = px.line(df, x="index", y="value", title=f"{title}")
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter(x=df["index"], y=df["value"], mode=mode, name="Wartość", line=dict(color='blue'), marker=marker_cfg)

    # Dodatek dla peaks
    if marker_peaks is not None:
        # Filtruj indeksy w zakresie
        valid_peaks = marker_peaks[(marker_peaks >= 0) & (marker_peaks < len(signal_real))]
        if len(valid_peaks) > 0:
            peaks_values = signal_real[valid_peaks]
            # Trójkąty dla wartości
            fig.add_scatter(x=valid_peaks, y=peaks_values, mode='markers', name="Peaks", marker=dict(symbol='triangle-up', size=10, color='rgba(0,0,0,0)', line=dict(color='red', width=1)))

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

def plot_symbols ( symbols : NDArray[ np.complex128 ] , title : str = "Symbole" ) -> None :
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

def complex_symbols_v0_1_6 ( complex_symbols : np.ndarray , title : str = "Konstelacja na płaszczyźnie zespolonej" ) -> None :
    """
    Ostateczna, niezniszczalna, piękna wersja.
    Działa na BPSK, QPSK, 16-QAM, APSK… na wszystkim.
    Zero błędów JSON, zero NumPy 2.0 issues.
    """
    z = np.asarray(complex_symbols, dtype=complex).ravel()

    # --- Wykrywanie unikalnych (idealnych) symboli ---
    rounded = np.round(z.real, 6) + 1j * np.round(z.imag, 6)
    unique = np.unique(rounded)

    # --- Przypisanie każdego symbolu do najbliższego idealnego ---
    idx = np.argmin(np.abs(z[:, np.newaxis] - unique[np.newaxis, :]), axis=1)
    labels = unique[idx]

    # --- Czytelne formatowanie liczb zespolonych ---
    def fmt(c: complex) -> str:
        re = f"{c.real:.4f}".rstrip("0").rstrip(".")
        if re == "" or re in ("-", "+"): re = "0"
        im = f"{abs(c.imag):.4f}".rstrip("0").rstrip(".")
        if im == "" or im == "0": 
            return re
        sign = "-" if c.imag < 0 else "+"
        return f"{re}{sign}{im}j"

    label_str = [fmt(c) for c in labels]
    unique_str = [fmt(c) for c in unique]

    # --- Główny wykres – teraz bezpiecznie z real/imag ---
    fig = px.scatter(
        x=z.real,                 # ← tylko float!
        y=z.imag,                 # ← tylko float!
        color=label_str,
        symbol=label_str,
        color_discrete_sequence=["#FFD700", "#FFEA00", "#CCFF00", "#AAFF00", "#99FF00", "#77FF00"],
        symbol_sequence=["circle", "square", "diamond", "x", "cross", "star"],
        opacity=0.82,
        title=f"<b>{title}</b>",
        labels={"x": "Re", "y": "Im"},
        width=900,
        height=900
    )

    # --- Idealne punkty ---
    fig.add_scatter(
        x=unique.real,
        y=unique.imag,
        mode="markers+text",
        marker=dict(size=34, color="#FFD700", line=dict(width=5, color="black")),
        text=[f"<b>{s}</b>" for s in unique_str],
        textposition="middle center",
        textfont=dict(color="black", size=14, family="Arial Black"),
        name="Idealne"
    )

    # --- Automatyczne marginesy (NumPy 2.0 safe) ---
    margin = 0.3
    if len(z) > 1:
        margin = max(0.3, 0.15 * (np.ptp(z.real) + np.ptp(z.imag)))

    fig.update_xaxes(
        range=[z.real.min() - margin, z.real.max() + margin],
        zeroline=True, zerolinewidth=3, zerolinecolor="#555",
        gridcolor="#222", linecolor="#666"
    )
    fig.update_yaxes(
        range=[z.imag.min() - margin, z.imag.max() + margin],
        zeroline=True, zerolinewidth=3, zerolinecolor="#555",
        scaleanchor="x", scaleratio=1,
        gridcolor="#222", linecolor="#666"
    )

    # --- Styl czarno-złoty ---
    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#000000",
        font=dict(color="#CCCCCC", size=14),
        title_x=0.5,
        legend=dict(
            title="Symbole",
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="#555",
            borderwidth=2,
            font=dict(color="#FFD700", size=12)
        )
    )

    # --- Złoty okrąg jednostkowy (jeśli ma sens) ---
    if np.max(np.abs(z)) <= 3.0:
        fig.add_shape(
            type="circle",
            x0=-1, y0=-1, x1=1, y1=1,
            xref="x", yref="y",
            line_color="#FFD700",
            line_width=2,
            opacity=0.4
        )

    fig.show()

def complex_waveform_v0_1_6 ( signal_complex : NDArray[ np.complex128 ] , title : str = "Sygnał zespolony", marker_squares : bool = False , marker_peaks : Optional[ NDArray[ np.uint32 ] ] = None ) -> None :
    """
    Rozszerzona wersja funkcji complex_waveform z dodatkowym parametrem marker_peaks.
    Jeśli marker_peaks zostanie przekazany (np.ndarray z indeksami), peaks zostaną zaznaczone trójkątami na wykresie.

    Parametry:
    - signal_complex: NDArray[np.complex128] (zespolony)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na wszystkich próbkach
    - marker_peaks: Optional[np.ndarray] — indeksy próbek, gdzie zaznaczyć trójkąty (rozmiar taki sam jak marker_squares)
    """
    if not np.iscomplexobj(signal_complex):
        raise ValueError("Wejściowy sygnał musi być zespolony NDArray[np.complex128]")

    df = pd.DataFrame({"index": np.arange(len(signal_complex)), "real": signal_complex.real, "imag": signal_complex.imag})

    if marker_squares:
        mode_real = 'lines+markers'
        mode_imag = 'lines+markers'
        marker_real_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='blue', width=1))
        marker_imag_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='orange', width=1))
    else:
        mode_real = 'lines'
        mode_imag = 'lines'
        marker_real_cfg = None
        marker_imag_cfg = None

    fig = px.line(df, x="index", y="real", title=f"{title}")
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter(x=df["index"], y=df["real"], mode=mode_real, name="I (real)", line=dict(color='blue'), marker=marker_real_cfg)
    fig.add_scatter(x=df["index"], y=df["imag"], mode=mode_imag, name="Q (imag)", line=dict(color='green', dash='dash'), marker=marker_imag_cfg)

    # Dodatek dla peaks
    if marker_peaks is not None:
        # Filtruj indeksy w zakresie
        valid_peaks = marker_peaks[(marker_peaks >= 0) & (marker_peaks < len(signal_complex))]
        if len(valid_peaks) > 0:
            peaks_real = signal_complex[valid_peaks].real
            peaks_imag = signal_complex[valid_peaks].imag
            # Trójkąty dla I (real)
            fig.add_scatter(x=valid_peaks, y=peaks_real, mode='markers', name="Peaks I", marker=dict(symbol='triangle-up', size=10, color='rgba(0,0,0,0)', line=dict(color='red', width=1)))
            # Trójkąty dla Q (imag)
            fig.add_scatter(x=valid_peaks, y=peaks_imag, mode='markers', name="Peaks Q", marker=dict(symbol='triangle-up', size=10, color='rgba(0,0,0,0)', line=dict(color='purple', width=1)))

    fig.update_layout(
        xaxis_title="Numer próbki",
        yaxis_title="Amplituda",
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    fig.show()


def flat_tensor_v0_1_18 (
    flat_tensor : torch.Tensor | NDArray[ np.complex64 ] | NDArray[ np.complex128 ] ,
    title : str = "Flat tensor TxSamples" ,
    marker_squares : bool = False ,
    marker_idx : Optional[ NDArray[ np.uint32 ] ] = None ,
    marker_peaks : Optional[ NDArray[ np.int_ ] ] = None
) -> None :
    """
    Rysuje 1D flat tensor zapisany przez TxSamples_v0_1_18.save_frames2flat_tensor.

    Obsługiwane wejścia:
    - torch.Tensor o kształcie [seq_len] i dtype zespolonym
    - np.ndarray o kształcie [seq_len] i dtype zespolonym
    """
    if isinstance ( flat_tensor , torch.Tensor ) :
        tensor_np = flat_tensor.detach ().cpu ().numpy ()
    else :
        tensor_np = np.asarray ( flat_tensor )

    if tensor_np.size == 0 :
        raise ValueError ( "Wejściowy flat_tensor jest pusty." )
    if tensor_np.ndim != 1 :
        raise ValueError ( "flat_tensor musi być tensorem 1D o kształcie [seq_len]." )
    if not np.iscomplexobj ( tensor_np ) :
        raise ValueError ( "flat_tensor musi zawierać wartości zespolone." )

    resolved_marker_idx = marker_idx if marker_idx is not None else marker_peaks

    complex_waveform_v0_1_6 (
        signal_complex = np.asarray ( tensor_np , dtype = np.complex64 ) ,
        title = f"{title} shape={tuple(tensor_np.shape)}" ,
        marker_squares = marker_squares ,
        marker_peaks = resolved_marker_idx
    )


def y_train_tensor_as_flat_tensor_v0_1_18 (
    y_train_tensor : torch.Tensor | NDArray[ np.complex64 ] | NDArray[ np.complex128 ] ,
    title : str = "Flat tensor from y_train_tensor" ,
    marker_squares : bool = False ,
    marker_peaks : Optional[ NDArray[ np.int_ ] ] = None
) -> None :
    """
    Zamienia zapisany y_train_tensor RxSamples_v0_1_18 na 1D flat tensor
    bez zmiany długości sygnału, a następnie wyświetla go funkcją
    flat_tensor_v0_1_18.

    Obsługiwane wejścia:
    - torch.Tensor o kształcie [seq_len] i dtype zespolonym
    - np.ndarray o kształcie [seq_len] i dtype zespolonym
    """
    if isinstance ( y_train_tensor , torch.Tensor ) :
        tensor = y_train_tensor.detach ().cpu ()
    else :
        tensor = torch.as_tensor ( np.asarray ( y_train_tensor ) )

    if tensor.numel () == 0 :
        raise ValueError ( "Wejściowy y_train_tensor jest pusty." )
    if tensor.ndim != 1 :
        raise ValueError ( "y_train_tensor musi być tensorem 1D o kształcie [seq_len]." )
    if not torch.is_complex ( tensor ) :
        raise ValueError ( "y_train_tensor musi zawierać wartości zespolone." )

    flat_tensor = tensor.to ( torch.complex64 )
    flat_tensor_v0_1_18 (
        flat_tensor = flat_tensor ,
        title = title ,
        marker_squares = marker_squares ,
        marker_peaks = marker_peaks
    )


def samples_and_tensor (
    samples : NDArray[ np.complex128 ] ,
    y_train_tensor : torch.Tensor | NDArray[ np.complex64 ] | NDArray[ np.complex128 ] ,
    tensor_m : int ,
    title : str
) -> None :
    if samples.size == 0 :
        raise ValueError ( "Wejściowe samples jest puste." )
    if samples.ndim != 1 :
        raise ValueError ( "samples musi być tablicą 1D." )
    if not np.iscomplexobj ( samples ) :
        raise ValueError ( "samples musi zawierać wartości zespolone." )

    if isinstance ( y_train_tensor , torch.Tensor ) :
        tensor_np = y_train_tensor.detach ().cpu ().numpy ()
    else :
        tensor_np = np.asarray ( y_train_tensor )

    if tensor_np.size == 0 :
        raise ValueError ( "Wejściowy y_train_tensor jest pusty." )
    if not np.iscomplexobj ( tensor_np ) :
        raise ValueError ( "y_train_tensor musi zawierać wartości zespolone." )

    tensor_flat = np.asarray ( tensor_np , dtype = np.complex64 ).reshape ( -1 )
    tensor_valid = tensor_flat * tensor_m
    tensor_x = np.arange ( tensor_valid.size )

    fig = px.line ( title = f"{title} | samples={samples.size} tensor={tensor_flat.size} tensor_m={tensor_m}" )
    fig.add_scatter ( x = np.arange ( samples.size ) , y = samples.real , mode = 'lines' , name = 'samples I (real)' , line = dict ( color = 'blue' ) )
    fig.add_scatter ( x = np.arange ( samples.size ) , y = samples.imag , mode = 'lines' , name = 'samples Q (imag)' , line = dict ( color = 'green' , dash = 'dash' ) )
    fig.add_scatter ( x = tensor_x , y = tensor_valid.real , mode = 'lines+markers' , name = 'tensor I (real)' , line = dict ( color = 'red' ) , marker = dict ( symbol = 'circle', size = 6 ) )
    fig.add_scatter ( x = tensor_x , y = tensor_valid.imag , mode = 'lines+markers' , name = 'tensor Q (imag)' , line = dict ( color = 'orange' , dash = 'dot' ) , marker = dict ( symbol = 'x', size = 7 ) )

    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( orientation = "h" , yanchor = "bottom" , y = 1.02 , xanchor = "center" , x = 0.5 ) ,
        height = 550
    )
    fig.show ()


def tensor_waveform_v0_1_16 (
    signal_tensor : torch.Tensor | NDArray[ np.float32 ] ,
    title : str = "Tensor RxSamples" ,
    marker_squares : bool = False ,
    marker_peaks : Optional[ NDArray[ np.int_ ] ] = None ,
    frame_idx : Optional[ int ] = None
) -> None :
    """
    Rysuje zawartość tensora I/Q w formie analogicznej do complex_waveform_v0_1_6.

    Obsługiwane wejścia:
    - [ batch , 2 , seq_len ]
    - [ 2 , seq_len ]
    - [ seq_len , 2 ]

    Gdy przekazano tensor 3D i `frame_idx is None`, wszystkie ramki są łączone
    w jeden ciąg czasowy, zachowując kolejność batchy.
    """
    if isinstance ( signal_tensor , torch.Tensor ) :
        tensor_np = signal_tensor.detach ().cpu ().numpy ()
    else :
        tensor_np = np.asarray ( signal_tensor )

    if tensor_np.size == 0 :
        raise ValueError ( "Wejściowy tensor jest pusty." )

    if tensor_np.ndim == 3 :
        if tensor_np.shape[ 1 ] != 2 :
            raise ValueError ( "Tensor 3D musi mieć układ [batch, 2, seq_len]." )
        if frame_idx is not None :
            if frame_idx < 0 or frame_idx >= tensor_np.shape[ 0 ] :
                raise ValueError ( f"frame_idx poza zakresem: {frame_idx=}, batch={tensor_np.shape[0]}" )
            iq_samples = tensor_np[ frame_idx ].T
            resolved_title = f"{title} frame={frame_idx} shape={tuple(tensor_np.shape)}"
        else :
            iq_samples = np.transpose ( tensor_np , ( 0 , 2 , 1 ) ).reshape ( -1 , 2 )
            resolved_title = f"{title} shape={tuple(tensor_np.shape)}"
    elif tensor_np.ndim == 2 :
        if tensor_np.shape[ 0 ] == 2 :
            iq_samples = tensor_np.T
        elif tensor_np.shape[ 1 ] == 2 :
            iq_samples = tensor_np
        else :
            raise ValueError ( "Tensor 2D musi mieć układ [2, seq_len] albo [seq_len, 2]." )
        resolved_title = f"{title} shape={tuple(tensor_np.shape)}"
    else :
        raise ValueError ( "Wejściowy tensor musi mieć 2 albo 3 wymiary." )

    signal_complex = iq_samples[ : , 0 ].astype ( np.float32 ) + 1j * iq_samples[ : , 1 ].astype ( np.float32 )
    complex_waveform_v0_1_6 (
        signal_complex = signal_complex ,
        title = resolved_title ,
        marker_squares = marker_squares ,
        marker_peaks = marker_peaks
    )

def prt_frames ( frames : list[ "packet.RxFrame_v0_1_18" ] , limit : int = len ( "packet.BARKER13_BITS" ) ,  ) -> None :
    for frame in frames :
        frame_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( np.concatenate ( [ frame.header_bpsk_symbols[ : : modulation.SPS ] , frame.packet.bpsk_symbols[ : : modulation.SPS ] ] ) )
        print ( f"{ frame_bits.size=}, {frame.frame_start_abs_idx=}, {frame_bits[ : limit ]=}" )