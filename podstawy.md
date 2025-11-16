###Czy samples odbieranych z Pluto zawierają informację o widmie RF?
Analiza próbek odbieranych z Pluto (np. via scipy.signal.welch) pozwala obliczyć PSD (Power Spectral Density) i OBW (Occupied Bandwidth) w domenie cyfrowej, co jest matematycznie ekwiwalentne pomiarowi fizycznego analizatora widma RF. Pokażę to krok po kroku, używając scipy, numba do optymalizacji, pandas do danych i plotly.express do wizualizacji.

####Próbkowanie i twierdzenie Nyquista:
W PlutoSDR (pyadi-iio), sygnał RF na wejściu RX (np. wokół rx_lo=2.9 GHz) jest filtrowany (rx_rf_bandwidth=1e6 Hz), downkonwertowany do baseband (0 Hz centrum) i próbkowany z sample_rate=3e6 Hz. Próbki to liczby zespolone I (in-phase) + jQ (quadrature), które reprezentują amplitudę i fazę sygnału w czasie. Według twierdzenia Nyquista-Shannona, jeśli fs >= 2 * max_freq (tu fs=3e6 > 2*BW=2e6), próbki wiernie odtwarzają oryginalne widmo do ±fs/2 (±1.5 MHz wokół centrum).
Ekwiwalencja cyfrowa: Fizyczny analizator widma RF (np. Keysight czy Rohde & Schwarz) też wewnętrznie digitalizuje sygnał i liczy FFT (Fast Fourier Transform) do PSD. W Pythonie robisz to samo: scipy.signal.welch to uśredniona FFT z oknem (np. Hann), co daje PSD bez hardware'u. Dla BPSK z RRC (beta=0.35, SPS=4), widmo ma kształt sinc-like z lobami bocznymi – offset freq przesunie pik, a fazowy poszerzy, co widać w PSD z próbek.
W praktyce: Jeśli masz offset freq >10 kHz (typowy w satelitarnym z Dopplera), PSD z próbek pokaże przesunięty pik mocy, co pozwala skompensować via scipy.signal.hilbert (estymacja fazy) i numba do szybkiej korekcji przed TX.

####Jak to działa w kodzie (bez fizycznego analizatora)?
Próbki z sdr.rx() (bufor 32768) to surowy sygnał po ADC – zawierają wszystkie częstotliwości w pasmie. Welch transformuje je do domeny częstotliwościowej:
- Rozbija na segmenty (nperseg), aplikuje FFT.
- Uśrednia, by zmniejszyć szum (noverlap).
- Wynik: f (częstotliwości) i Pxx (moc na Hz), co daje OBW via kumulacyjna suma mocy (jak w mojej poprzedniej propozycji).
To nie symulacja – to rzeczywista analiza! Różnica od analizatora RF: SDR ma mniejszą dynamikę (np. 12-bit ADC w Pluto), ale dla Twojego BW=1e6 Hz i F_S=3e6 to wystarczy do estymacji OBW ~1 MHz (Rs=750 kHz * (1+beta)=~1.01 MHz dla BPSK).
Przykład: Jeśli samples z CSV/logs mają offset, PSD pokaże poszerzone widmo – użyj tego do tuningu w corrections.py.

####Ograniczenia i porównanie do fizycznego analizatora
Zalety cyfrowej analizy: Szybka, tania, integruje się z korekcją offsetu (np. hilbert do fazy, numba do mnożenia exp(1j*phase_offset)). Dla satelitarnego BPSK, możesz to robić w pętli real-time bez dodatkowego sprzętu.
Wady: SDR ma szum kwantyzacji (niższa rozdzielczość niż analizator RF z 16-bit ADC), i nie widzi poza pasmem (BW=1e6 Hz). Jeśli offset freq > fs/2 (1.5 MHz), aliasing zniekształci PSD – wtedy użyj fizycznego analizatora do weryfikacji.
Weryfikacja: Porównaj z symulacją w scipy (wygeneruj BPSK samples z offsetem) – PSD będzie identyczne jak z rzeczywistych próbek.