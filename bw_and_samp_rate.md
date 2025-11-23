Poniżej praktyczne reguły, formuły i konkretne rekomendacje — jak dobrać optymalne sample rate i bandwidth (RF) dla nadajnika na F_C = 2.9 GHz

# Kluczowe założenia i wzory

1. Symbol rate Rs = Fs / sps (gdzie Fs to sampling rate cyfrowy, sps = samples per symbol)
1. Przy RRC (roll-off β) przybliżone całkowite zajęte pasmo (dwustronne) ≈ Rs * (1 + β)
Innymi słowy: jeśli nadajesz BPSK z Rs, to szerokość sygnału ≈ Rs*(1+β)
1. Konsekwencja: Fs powinno być równe sps * Rs, a RF/BW ustaw na nieco większe niż Rs*(1+β) (z marginesem)

# Przykład z twoich ustawień (konkretne liczby)

- `sps = 4`, `Fs = 3 000 000` → `Rs = 750 000` (symbols/s)
- `β = 0.35` → sygnał zajmuje ≈ `Rs*(1+β) = 750k * 1.35 ≈ 1.0125 MHz`
Wniosek: ustawienie `BW = 1e6` i `Fs = 3e6` jest bardzo bliskie — dodaj mały margines (np. 10–20 %) dla wygody i filtrów: `BW ≈ 1.1–1.2 MHz`

# Praktyczne rekomendacje (reguły)

1. Dobierz `sps` w przedziale 4–8 (4 to wygodny kompromis wydajnościowy).
1. Wybierz `Fs = sps * Rs` (przy projektowaniu) lub odwrotnie: jeśli chcesz Fs ograniczone przez SDR, policz `Rs = Fs/sps`.
1. Oblicz teoretyczne pasmo: BW_sig ≈ Rs*(1+β).
1. Ustaw RF/decyzator BW w SDR na BW_sdr ≈ 1.1…1.3 * BW_sig (10–30% zapasu), ale nie większe niż to co pozwala sprzęt (często BW <= Fs).
1. Ustawienie BW RF mniejsze od sygnału spowoduje ucięcia; za duże — większy szum wejściowy. Znajdź balans.

# Uwagi dotyczące ADALM‑Pluto (praktyka)

1. Pluto ma parametry: sample rate (cyfrowy) i RF bandwidth (filtr frontend). Typowe kombinacje: Fs = 3e6, BW = 1e6 — to pasuje dla Rs≈750k sps=4.
1. Ustaw tx_bandwidth i tx_sample_rate tak, by front-end nie obcinał sygnału; najlepiej BW >= sygnał_occupied.
1. Dla rx: podobnie — ustaw rx_bandwidth na >= sygnału; decymacja/filtracja w DSP można zrobić potem.

# Jak empirycznie dobrać (procedura)

1. Zmierz PSD sygnału (np. Welch) i wyznacz pasmo zajęte (np. od częstotliwości, gdzie moc spada poniżej -3 dB od max albo użyj kryterium energetycznego np. 99% energii).
1. Oblicz `Rs_est = BW_measured / (1 + β)`
1. Wybierz `sps` np. 4, policz `Fs = Rs_est * sps` (zaokrąglij do wspieranego przez SDR)
1. Ustaw `BW_sdr = BW_measured * 1.1..1.3` i Fs zgodnie z krokiem 3
1. Przetestuj: sprawdź czy nie ma aliasingu i czy sygnał nie jest obcinany; sprawdź eye / BER.
1. Kod pomocniczy (Python) — znaleźć szerokość zajętą (np. 99% energii)
`import numpy as np`
`from scipy.signal import welch`

`def occupied_bandwidth(sig, fs, pct=0.99, nperseg=4096):`
`    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)`
`    Pcum = np.cumsum(Pxx)`
`    Ptotal = Pcum[-1]`
`    low_idx = np.searchsorted(Pcum, (1 - pct)/2 * Ptotal)`
`    high_idx = np.searchsorted(Pcum, (1 + pct)/2 * Ptotal)`
`    bw = f[high_idx] - f[low_idx]`
`    return bw, f[low_idx], f[high_idx]`
``
`# przykład użycia`
`bw, f_lo, f_hi = occupied_bandwidth(tx_packet_upsampled, fs=3e6, pct=0.99)`
`print(f"occupied BW ~ {bw/1e6:.3f} MHz from {f_lo/1e6:.6f} to {f_hi/1e6:.6f} MHz")`

Jeśli wynik ~1.0 MHz → zastosuj opisane wyżej ustawienia.

#Dodatkowe praktyczne wskazówki

1. Zapas (guard):
- filtry front-end mają nachylenie — daj 10–30% marginesu
- jeśli planujesz pasma kanałów obok siebie, zostaw większy odstęp
1. Jeśli chcesz maksymalnie zredukować szum: ustaw `BW` co najwyżej niezbędne, ale nie mniejsze niż sygnał
1. Pamiętaj o normalizacji amplitudy (unikaj clippingu). Przy Pluto typowy pełny zakres cyfrowy wynosi `±2^14`; skaluj sygnał przed wysłaniem
1. Jeśli chcesz obsługiwać szersze pasmo/wyższe `Rs`, rozważ zwiększenie `sps` (np. 8) lub zwiększenie `Fs`, ale sprawdź ograniczenia sprzętowe (maks. `Fs` i dostępne `BW`)

#Szybkie zestawienie rekomendowane wartości (dla twojego przypadku)

Center freq: 2.9 GHz (fixed)
sps: 4 (domyślnie)
Fs (sample rate): 3.0 MHz (lub najbliższy wspierany)
Rs = Fs/sps = 750 kbaud
β (RRC): 0.35 → sygnał BW ≈ 750k * 1.35 ≈ 1.01 MHz
RF bandwidth ustaw: 1.1–1.2 MHz (albo 1.5 MHz jeśli chcesz większy zapas)
Jeśli chcesz mniejszy szum i wiesz że sygnał węższy: ustaw RF BW bliżej 1.05 MHz, ale sprawdź czy nie obcina.