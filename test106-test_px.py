import numpy as np
import os
import pandas as pd
from scipy.signal import correlate, find_peaks
import plotly.express as px

from modules import ops_file , plot

script_filename = os.path.basename ( __file__ )

tx_samples_barker13_filename = "logs/tx_samples_barker13_clipped_74.csv"
rx_samples_filename = "logs/rx_samples_32768.csv"

tx_samples_barker13 = ops_file.open_csv_and_load_np_complex128 ( tx_samples_barker13_filename )
rx_samples = ops_file.open_csv_and_load_np_complex128 ( rx_samples_filename )

plot.plot_complex_waveform_v2 ( rx_samples , script_filename )

import plotly.express as px

samples = rx_samples
if not np.iscomplexobj ( samples ) :
    raise ValueError ( "Wejściowy sygnał musi być zespolony (np.ndarray typu complex)." )

# Tworzenie DataFrame do wykresu
df = pd.DataFrame ( { "index": np.arange ( len ( samples ) ) , "real": samples.real , "imag": samples.imag } )
fig = px.line(df, x="index", y=["real", "imag"], labels={"variable": f"{script_filename} {samples.size=}"} )

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