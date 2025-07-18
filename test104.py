import numpy as np
import pandas as pd
import plotly.express as px

# --- SYMULACJA OBRACAJĄCEGO SIĘ SYGNAŁU ---
N = 200
f_offset = 0.02  # symulowany dryf fazy
symbol = 1 + 1j  # stały symbol
n = np.arange(N)

# Obracający się sygnał
phase_drift = np.exp(1j * 2 * np.pi * f_offset * n)
received = symbol * phase_drift

# --- ESTYMACJA FAZY ---
phase = np.unwrap(np.angle(received))
# Dopasowanie prostej do fazy: y = ax + b
coeffs = np.polyfit(n, phase, 1)
estimated_drift = coeffs[0]  # współczynnik kierunkowy (czyli dryf fazy)

# --- KOREKCJA OBRACAJĄCEGO SIĘ SYGNAŁU ---
correction = np.exp(-1j * estimated_drift * n)
corrected = received * correction

# --- TWORZENIE TABELI DO WYKRESU ---
df = pd.DataFrame({
    'I': np.concatenate([received.real, corrected.real]),
    'Q': np.concatenate([received.imag, corrected.imag]),
    'Typ': ['Przed korekcją'] * N + ['Po korekcji'] * N
})

# --- KONSTELACJA IQ ---
fig = px.scatter(df, x='I', y='Q', color='Typ',
                 title='Konstelacja IQ przed i po korekcji obrotu fazy',
                 labels={'I': 'In-phase (I)', 'Q': 'Quadrature (Q)'},
                 width=600, height=600)
fig.update_traces(marker=dict(size=6))
fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
fig.show()
