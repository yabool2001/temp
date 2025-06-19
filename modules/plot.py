import numpy as np
import pandas as pd
import plotly.express as px

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