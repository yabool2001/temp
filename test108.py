import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =============================================
# Parametry
# =============================================
N = 6000
EbN0_dB = 12

# =============================================
# Symbole QPSK + szum
# =============================================
constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
bits = np.random.randint(0, 4, N)
symbols = constellation[bits]

EsN0 = 10**(EbN0_dB / 10) * 2
sigma = np.sqrt(1 / (2 * EsN0))
noise = sigma * (np.random.randn(N) + 1j * np.random.randn(N))
received = symbols + noise

labels = np.array(['00', '01', '11', '10'])[bits]

# =============================================
# WYKRES – CZARNO-ZŁOTY STYL
# =============================================
fig = px.scatter(
    x=received,                               # bezpośrednio liczby zespolone!
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

# =============================================
# Czarno-szare, ultra eleganckie tło i osie
# =============================================
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

fig.show()