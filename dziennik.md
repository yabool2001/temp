# Dziennik projektu

prompt

Sprawdziłem Twoją hipotezę na swój sposób, tj.:
# Opuszczam pierwsze 20 sampli po samplu #40139 (first_symbol_abs_idx) to jest: 20 / modulation.SPS = 5 symboli, żeby ominąć początkowe artefakty z zimnego startu LSTM i
# 1. Decymuję kolejne 100 sampli z każdym racjonalnym offsetem pomimo, że znam dokładnie to co wysłałem do demodulatora AI i wiem precyzyjnie, gdzie jest pierwszy sample, a gdzie idealny modulation.SPS // 2 = 2. Ale chcę sprawdzić wszytskie opcja offset jak zadziałają.
# 2. Porównuję wynik z idealnymi symbolami BPSK, które powinny być dokładnie takie same, przy jakims offset i idealna liczba symboli i liczę ile jest niezgodności  między idealnymi symbolami a tymi z AI, żeby mieć miarę jakości demodulacji AI. To są wyniki:
perfect_offset=0: Po AI, w pierwszych 100 decymowanych symbolach, jest 22 niezgodności w porównaniu do idealnych symboli BPSK!
perfect_offset=1: Po AI, w pierwszych 100 decymowanych symbolach, jest 23 niezgodności w porównaniu do idealnych symboli BPSK!
perfect_offset=2: Po AI, w pierwszych 100 decymowanych symbolach, jest 28 niezgodności w porównaniu do idealnych symboli BPSK!
perfect_offset=3: Po AI, w pierwszych 100 decymowanych symbolach, jest 53 niezgodności w porównaniu do idealnych symboli BPSK!
Załączam również aktualny kod i obrazki z symbolami po decymacji po 5 samplach barkera13
Wynik nie napawa optymizmem. Wychodzi na to, że próbujemy działać na chybił trafił bez głębokiej analizy i root cause analysis. A bardzo bym chciał żebyś solidnie przejrzał załączony kod.

## 2026.06.09 Teraz mam fatalne wyniki uczenia
Epoka [01/15] | Błąd EVM (MSE): 0.47239 | Czas epoki: 55.06 s
Epoka [02/15] | Błąd EVM (MSE): 0.47119 | Czas epoki: 53.93 s
Epoka [03/15] | Błąd EVM (MSE): 0.47103 | Czas epoki: 53.65 s
Epoka [04/15] | Błąd EVM (MSE): 0.47085 | Czas epoki: 53.69 s
Epoka [05/15] | Błąd EVM (MSE): 0.47099 | Czas epoki: 53.70 s
Epoka [06/15] | Błąd EVM (MSE): 0.47101 | Czas epoki: 53.57 s

## 2026.06.04 wprowadziłem adaptacyjny learning rate w test129-training.py

```python
# Najlepsze rozwiązanie: automatyczny scheduler zmniejszający LR przy stagnacji
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau ( optimizer , mode = 'min' , factor = 0.1 , patience = 2 )

    # Aktualizacja schedulera: jeśli wymacana dolina jest płaska przez 2 epoki, tnie bazowy LR!
    scheduler.step ( avg_loss )
```
I jest lepszy końcowy efekt
Epoka [01/15] | Błąd EVM (MSE): 0.12048 | Czas epoki: 113.90 s
Epoka [02/15] | Błąd EVM (MSE): 0.11828 | Czas epoki: 112.01 s
Epoka [03/15] | Błąd EVM (MSE): 0.11811 | Czas epoki: 111.44 s
Epoka [04/15] | Błąd EVM (MSE): 0.11800 | Czas epoki: 111.92 s
Epoka [05/15] | Błąd EVM (MSE): 0.11798 | Czas epoki: 111.64 s
Epoka [06/15] | Błąd EVM (MSE): 0.11792 | Czas epoki: 111.55 s
Epoka [07/15] | Błąd EVM (MSE): 0.11795 | Czas epoki: 110.93 s
Epoka [08/15] | Błąd EVM (MSE): 0.11802 | Czas epoki: 110.98 s
Epoka [09/15] | Błąd EVM (MSE): 0.11799 | Czas epoki: 111.17 s
Epoka [10/15] | Błąd EVM (MSE): 0.11796 | Czas epoki: 112.00 s
Epoka [11/15] | Błąd EVM (MSE): 0.11804 | Czas epoki: 112.82 s
Epoka [12/15] | Błąd EVM (MSE): 0.11778 | Czas epoki: 110.91 s
Epoka [13/15] | Błąd EVM (MSE): 0.11803 | Czas epoki: 111.73 s
Epoka [14/15] | Błąd EVM (MSE): 0.11802 | Czas epoki: 111.17 s
Epoka [15/15] | Błąd EVM (MSE): 0.11793 | Czas epoki: 112.30 s

Model zostal zapisany jako 

## 2026.05.26 testy nowej funkcji clip_samples_and_create_tensor_4_training która bazuje na wklejaniu sampli a nie symboli i jest wielki progres
Epoka [01/15] | Błąd EVM (MSE): 0.12287 | Czas epoki: 114.12 s
Epoka [02/15] | Błąd EVM (MSE): 0.11848 | Czas epoki: 113.07 s
Epoka [03/15] | Błąd EVM (MSE): 0.11810 | Czas epoki: 115.24 s
Epoka [04/15] | Błąd EVM (MSE): 0.11819 | Czas epoki: 110.82 s
Epoka [05/15] | Błąd EVM (MSE): 0.11821 | Czas epoki: 110.80 s
Epoka [06/15] | Błąd EVM (MSE): 0.11806 | Czas epoki: 110.98 s
Epoka [07/15] | Błąd EVM (MSE): 0.11818 | Czas epoki: 110.55 s
Epoka [08/15] | Błąd EVM (MSE): 0.11817 | Czas epoki: 110.78 s
Epoka [09/15] | Błąd EVM (MSE): 0.11806 | Czas epoki: 110.65 s
Epoka [10/15] | Błąd EVM (MSE): 0.11795 | Czas epoki: 110.76 s
Epoka [11/15] | Błąd EVM (MSE): 0.11811 | Czas epoki: 112.09 s
Epoka [12/15] | Błąd EVM (MSE): 0.11797 | Czas epoki: 112.40 s
Epoka [13/15] | Błąd EVM (MSE): 0.11797 | Czas epoki: 113.66 s

## 2026.05.20 praca w Lipkowie
*   **`np.tensors_004`** zawiera 2 x 8 oryginalnych sampli wygenerowane przez `test134-tx_large_frames.py` i zapisanych przez `test134-rx_large_frames.py`. Sample były transmitowane przez radia połączone tłumikiem 30 dB. tx transmitował z `gain = -10.0`, a rx odbierał w trybie `slow_attack`.

## Na samym początku wyniki były super
Epoka [01/15] | Błąd EVM (MSE): 0.20830 | Czas epoki: 35.05 s
Epoka [02/15] | Błąd EVM (MSE): 0.21843 | Czas epoki: 35.90 s
Epoka [03/15] | Błąd EVM (MSE): 0.21799 | Czas epoki: 36.58 s
Epoka [04/15] | Błąd EVM (MSE): 0.21015 | Czas epoki: 36.40 s
Epoka [05/15] | Błąd EVM (MSE): 0.21007 | Czas epoki: 35.20 s
Epoka [06/15] | Błąd EVM (MSE): 0.21789 | Czas epoki: 35.29 s