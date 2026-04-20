# 📡 Dokumentacja Architektury: Kognitywny Demodulator AI-SDR (CVNN)

## 1. Wstęp i Abstrakt Projektu
Niniejszy dokument opisuje architekturę w pełni autonomicznego odbiornika warstwy fizycznej (L1) dla sygnałów radiowych. Rozwiązanie to opiera się na **CVNN (Complex-Valued Neural Network)** – głębokiej sieci neuronowej operującej natywnie na liczbach zespolonych. 

Model przyjmuje surowe, zaszumione próbki I/Q bezpośrednio z urządzenia Software-Defined Radio (np. PlutoSDR) i realizuje **ślepą demodulację (Blind Demodulation)** sygnału (np. BPSK). 

**Innowacja:** Architektura całkowicie eliminuje konieczność stosowania klasycznych bloków DSP (Digital Signal Processing). Nie wykorzystuje programowych pętli odzyskiwania nośnej (Costas Loop), filtrów dopasowanych (Root-Raised-Cosine), ani algorytmów synchronizacji symboli (Gardner Timing Recovery). Kompensacja potężnego dryfu częstotliwości (CFO) i redukcja szumu odbywają się wyłącznie poprzez nieliniowe operacje macierzowe na układzie GPU.

---

## 2. Przepływ Danych (Data Pipeline)

System operuje w paradygmacie **Fractionally Spaced Equalization** (Korekcja z nadpróbkowaniem) oraz **Gęstego Nadzoru (Dense Supervision)**.

*   **Brak Decymacji:** Wewnątrz sieci nie dochodzi do utraty rozdzielczości czasu. Stosunek próbek wejściowych do wyjściowych wynosi 1:1.
*   **Wejście:** Surowy wektor zespolony `[Batch, 1 Kanał, 4096 sampli]` (typ: `torch.complex64`). Sygnał obarczony jest wirującą fazą i szumem termicznym (AWGN). Przed wejściem do sieci nakładane jest algorytmiczne Cyfrowe AGC (Automatic Gain Control) normalizujące amplitudę okna.
*   **Wyjście:** Skorygowany wektor zespolony `[Batch, 4096]`, w którym faza została zablokowana, a wektory rzutowane są na twarde punkty konstelacji.

---

## 3. Topologia Sieci (`HardcoreComplexEqualizer`)

Przetwarzanie sygnału odbywa się rurociągowo (Pipeline) w trzech głównych etapach:

### Etap 1: Ekstrakcja Cech (Cyfrowy Filtr Dopasowany)
```python
self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding='same', dtype=torch.complex64)
```

Rola fizyczna: Warstwa splotowa działa jako inteligentny filtr adaptacyjny. Optymalizując wagi, sieć uczy się agregować energię użyteczną w czasie i odcinać szum spoza pasma, wygładzając interferencje międzysymbolowe (ISI).

Mechanika: Utrzymanie stride=1 i padding='same' gwarantuje przekazanie pełnej osi czasu do silnika rekurencyjnego. Rozszerza sygnał na 16 równoległych kanałów cech.