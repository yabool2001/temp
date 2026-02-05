# Wstęp
Odbiór i demodulacja sygnału znajdującego się poniżej poziomu szumu (tzw. "podszumowego" lub sub-noise) jest jak najbardziej możliwy i powszechnie stosowany, np. w systemach GPS czy komunikacji kosmicznej.

Oto główne techniki, które na to pozwalają:

# Korelacja i Filtrowanie Dopasowane (Matched Filtering)

To najczęstsza metoda, którą zresztą stosuję w skryptach (np. correlation.py).

## Jak to działa?
Jeśli odbiornik zna kształt sygnału (np. sekwencję synchronizacyjną, kod Barkera), może wykonać operację korelacji wzajemnej (cross-correlation) odebranego zaszumionego sygnału z tym wzorcem.

## Efekt
Operacja ta "kumuluje" energię sygnału w jednym punkcie (pik korelacji), podczas gdy szum (będący losowym) uśrednia się do niskich wartości. Dzięki temu pik sygnału może wystawać ponad szum, nawet jeśli sam przebieg w dziedzinie czasu jest w nim "utopiony".

## Zysk przetwarzania (Processing Gain):
Im dłuższa znana sekwencja, tym większy zysk energetyczny.

# Rozpraszanie Widma (DSSS - Direct Sequence Spread Spectrum)

Technika używana w GPS i CDMA. Sygnał użyteczny jest mnożony przez szybkozmienną sekwencję kodową (rozpraszającą), co powoduje, że staje się on bardzo szerokopasmowy i "chowa się" w szumie.

## Odbiór
Odbiornik "od-rozprasza" sygnał używając tej samej sekwencji, co przywraca oryginalny sygnał wąskopasmowy o dużej mocy, a szum tła (który nie pasuje do sekwencji) zostaje rozproszony i stłumiony.

# Uśrednianie (Averaging)

Jeśli sygnał jest powtarzalny (np. periodyczny pilot lub preambuła), można zsumować wiele jego powtórzeń.

Sygnał koherentny dodaje się "napięciowo" (liniowo).
Szum nieskorelowany dodaje się "mocowo" (pierwiastek z sumy kwadratów).
W rezultacie SNR (stosunek sygnału do szumu) rośnie wraz z liczbą uśrednień.
Czy to możliwe dla sygnału wąskopasmowego? Tak, ale z pewnymi zastrzeżeniami:

## W domenie czasu
Sygnał wąskopasmowy może mieć amplitudę mniejszą niż szum. Wtedy, aby go odebrać, musimy znać jego strukturę (np. piloty, sekwencje uczące) i użyć korelacji lub bardzo silnych kodów korekcyjnych (FEC - np. Turbo kody, LDPC).

## Filtrowanie
Jeśli "poziom szumu" oznacza szum w całym paśmie odbiornika (np. 1 MHz), a Twój sygnał zajmuje tylko 10 kHz, to zastosowanie wąskopasmowego filtra cyfrowego (o szerokości 10 kHz) wytnie większość szumu spoza pasma sygnału, drastycznie poprawiając SNR przed demodulacją.