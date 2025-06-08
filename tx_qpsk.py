# https://pysdr.org/content/pluto.html#transmitting

import adi
import numpy as np
import pandas as pd
import plotly.express as px

def plot_tx_waveform(samples):
    print("Rysuję wykres...")

    df = pd.DataFrame({
        "I": np.real(samples),
        "Q": np.imag(samples),
        "index": np.arange(len(samples))
    })

    fig = px.line(df, x="index", y="I", title="Sygnał Tx QPSK: I i Q")
    fig.add_scatter(x=df["index"], y=df["Q"], mode="lines", name="Q (imag)", line=dict(dash="dash"))
    fig.update_layout(
        xaxis_title="Numer próbki",
        yaxis_title="Amplituda",
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    fig.show()


num_symbols = 1000
x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
plot_tx_waveform ( samples )
pass