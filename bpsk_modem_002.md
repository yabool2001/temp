# Dokumentacja Architektury: Kognitywny Demodulator AI-SDR (CVNN)

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
self.conv1 = nn.Conv1d ( 1 , 16 , kernel_size = 7 , stride = 1 , padding = 'same' , dtype = torch.complex64 )
```

*   **Rola fizyczna:** Warstwa splotowa działa jako inteligentny filtr adaptacyjny. Optymalizując wagi, sieć uczy się agregować energię użyteczną w czasie i odcinać szum spoza pasma, wygładzając interferencje międzysymbolowe (ISI).
*   **Mechanika:** Utrzymanie stride=1 i padding='same' gwarantuje przekazanie pełnej osi czasu do silnika rekurencyjnego. Rozszerza sygnał na 16 równoległych kanałów cech.

### Etap 2: Silnik Śledzenia Fazy (Cyfrowe PLL)
```python
self.lstm = PureComplexLSTM ( input_size = 16 , hidden_size = 64 )
```

*   **Rola fizyczna:** Zastępuje sprzętową pętlę fazową (PLL). Wykorzystuje pamięć wewnętrzną do wyliczania pochodnej rotacji wektorów, dynamicznie odkręcając wirującą fazę CFO.
*   **Technika "Split-Activation":** Natywny moduł `nn.LSTM` we frameworkach korporacyjnych nie obsługuje natywnie w pełni algebry zespolonej dla akceleracji CUDA. Stworzono autorską komórkę.

1. *Mnożenie Macierzy:* Transformacje liniowe (obroty Eulera) wykonywane są natywnie przez `nn.Linear(dtype=torch.complex64)`.
2. *Rozszczepienie Aktywacji:* Zespolone funkcje trygonometryczne (`Sigmoid`, `Tanh`) powodują niestabilność matematyczną i błędy `NaN`. Sygnał jest ułamkowo rozszczepiany na osie Rzeczywistą i Urojoną, przepuszczany przez zawory nieliniowe jako float32, a następnie sprzętowo scalany przez `torch.complex()`. Gwarantuje to gładką propagację BPTT.

### Etap 3: Zespolony Decydent (Squelch)
```python
self.fc = nn.Linear(64, 1, dtype=torch.complex64)
```
*   **Rola fizyczna:** Zrzutowanie 64-wymiarowej pamięci z powrotem na fizyczną płaszczyznę konstelacji I/Q. Pełni funkcję "Inteligentnej Bramki Szumów" (Squelch) – podczas ciszy radiowej w eterze wygasza sztucznie wektor do zera absolutnego 0+0j.

---
## 4. Wyzwania Inżynierskie i Obejścia (Workarounds)

### 4.1. Ominięcie braku sprzętowego `mse_cuda` (Zero-Copy)
Środowisko CUDA (NVIDIA) nie posiada sprzętowej implementacji błędu średniokwadratowego (MSE) dla wektorów `ComplexFloat`. Aby zachować pełną prędkość na GPU, zastosowano rzutowanie w locie:
```python
loss = criterion(torch.view_as_real(predictions), torch.view_as_real(batch_y))
```

Karta graficzna traktuje płaszczyznę zespoloną jako dwuwymiarową siatkę (X, Y) typu zmiennoprzecinkowego. Euklidesowy dystans błędu EVM liczony jest natywnie bez narzutu na pamięć RAM.

### 4.2. Balansowanie Danych (Data-Centric AI)
Występowało zjawisko tzw. *Pułapki Leniwego Squelcha* – model optymalizował błąd rzucając ciągłe zera, ignorując trudną matematykę fazy (ponieważ w eterze dominowała cisza).
Zastosowano rygorystyczne balansowanie zbioru uczącego. Ramki składają się w ~90% z gęstych symboli użytecznych i jedynie w ~10% z ochronnego marginesu szumu. Zmusiło to AI do "otwarcia oka" sygnału.

### 4.3. Curriculum Learning i Stabilizacja
Ze względu na brak decymacji, graf rekurencyjny rozwijany jest na tysiące kroków dla pojedynczej klatki.

*   **Zastosowano twardy kaganiec gradientowy (`clip_grad_norm_`), chroniący przed eksplozją wag w czasie potężnych rotacji fazy.
*   **Trening odbywa się fazowo: najpierw sieć uczy się kompensacji CFO na wysokim SNR, a w drugiej fazie (Transfer Learning) dostraja warstwę splotową do walki z szumem amplitudowym (AWGN) przy zredukowanym `lr`.

---
## 5. Tryb Produkcyjny (Inference / Live SDR)
W zastosowaniu produkcyjnym na żywo, sieć ewoluje z symulatora do bezpołączeniowej "Czarnej Skrzynki":

1. **Odcięcie zależnosci:** Skrypty korelatora i pliki wzorcowe zostają całkowicie odcięte.
1. **Rurociąg Matrycowy:** Pętle w języku Python ulegają usunięciu. Setki tysięcy surowych próbek eteru z anteny zostają przeformatowane funkcją .reshape do pojedynczej macierzy 3D.
1. **One-Shot Execution:** Karta graficzna (np. RTX 5080) wciąga macierz naraz, wykonuje sprzętową demodulację podczas pojedynczego zrzutu (Forward-Pass).
1. **Dekodowanie Twarde:** Odtworzona konstelacja complex128 przepuszczana jest przez komparator (.real > 0), generując twardy strumień bitów gotowy do odczytu przez parser warstwy MAC.