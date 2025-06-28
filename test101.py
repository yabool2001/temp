import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parametry
N = 200
f_offset = 0.02  # obrót fazy
symbol = 1 + 1j
n = np.arange(N)
phase_drift = np.exp(1j * 2 * np.pi * f_offset * n)
received = symbol * phase_drift

I_vals = received.real
Q_vals = received.imag

# Tworzymy subploty: konstelacja + sinus i cosinus
fig = make_subplots(rows=1, cols=2, subplot_titles=["Konstelacja (I vs Q)", "I/Q w czasie"],
                    specs=[[{"type": "scatter"}, {"type": "scatter"}]])

# Inicjalizacja punktów
fig.add_trace(go.Scatter(x=[I_vals[0]], y=[Q_vals[0]], mode='markers',
                         marker=dict(size=12), name='IQ Punkt'), row=1, col=1)

fig.add_trace(go.Scatter(x=n[:1], y=I_vals[:1], mode='lines+markers', name='I(t)'), row=1, col=2)
fig.add_trace(go.Scatter(x=n[:1], y=Q_vals[:1], mode='lines+markers', name='Q(t)'), row=1, col=2)

# Tworzenie ramek animacji
frames = []
for i in range(1, N):
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[I_vals[i]], y=[Q_vals[i]], mode='markers',
                       marker=dict(size=12)),
            go.Scatter(x=n[:i], y=I_vals[:i], mode='lines+markers'),
            go.Scatter(x=n[:i], y=Q_vals[:i], mode='lines+markers')
        ],
        name=str(i)
    ))

# Ustawienia animacji
fig.update(frames=frames)

fig.update_layout(
    title="Animacja: Obracający się sygnał IQ oraz jego składowe sinus/cosinus",
    width=900,
    height=500,
    xaxis=dict(title='I'), yaxis=dict(title='Q', scaleanchor="x", scaleratio=1),
    xaxis2=dict(title='Numer próbki'), yaxis2=dict(title='Amplituda'),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[dict(label='Start', method='animate', args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}])]
    )]
)

fig.show()
