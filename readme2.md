Dane w formacie `complex128` zamiast `int16` w bibliotece PyADI-IIO dla PlutoSDR wynikają z tego, jak biblioteka przetwarza i zwraca dane z odbiornika (RX). Oto wyjaśnienie:

1. **Format danych w PyADI-IIO**:
   - Biblioteka PyADI-IIO, używana do sterowania urządzeniami Analog Devices (np. ADALM-Pluto), korzysta z biblioteki `libiio` do komunikacji z urządzeniem. Surowe dane z odbiornika PlutoSDR są próbkami I/Q (in-phase i quadrature), które w sprzęcie są w formacie 12-bitowym (reprezentowanym w `int16` z przesunięciem bitowym, jak np. `le:S12/16>>0`).
   - Jednak PyADI-IIO, wywołując metodę `sdr.rx()`, automatycznie konwertuje surowe dane I/Q na format `complex128` (128-bitowy typ zmiennoprzecinkowy, składający się z dwóch 64-bitowych liczb zmiennoprzecinkowych dla części rzeczywistej i urojonej). Jest to zrobione dla wygody użytkownika, aby dane były łatwiejsze do przetwarzania w Pythonie, szczególnie w bibliotekach takich jak NumPy, które dobrze radzą sobie z typami zmiennoprzecinkowymi.

2. **Dlaczego `complex128`?**:
   - **Wygoda i kompatybilność**: Format `complex128` jest standardem w Pythonie (NumPy) dla danych zespolonych, co ułatwia manipulację sygnałami I/Q w zastosowaniach DSP (przetwarzanie sygnałów cyfrowych). Umożliwia to bezpośrednie stosowanie operacji matematycznych bez konieczności ręcznej konwersji typów.
   - **Skalowanie i normalizacja**: PyADI-IIO automatycznie skaluje surowe dane `int16` (12-bitowe próbki z ADC) do wartości zmiennoprzecinkowych w zakresie [-1, 1] w formacie `complex128`. To eliminuje potrzebę ręcznego zarządzania skalą sygnału przez użytkownika.
   - **Abstrakcja**: PyADI-IIO została zaprojektowana tak, aby ukryć szczegóły niskopoziomowe `libiio`, takie jak format surowych danych (`int16`), i dostarczyć dane w bardziej przystępnym formacie dla aplikacji Pythonowych.

3. **Surowe dane w formacie `int16`**:
   - Sprzętowo, PlutoSDR dostarcza dane w formacie `int16` (12-bitowe próbki I/Q zapisywane w 16-bitowych słowach). Można to potwierdzić, korzystając z narzędzi takich jak `iio_info`, które pokazują format kanału jako `le:S12/16>>0` dla odbiornika (`cf-ad9361-lpc`).[](https://wiki.analog.com/resources/tools-software/linux-software/libiio/iio_readdev)
   - Jeśli chcesz uzyskać surowe dane w formacie `int16`, musisz bezpośrednio użyć API `libiio` w Pythonie (moduł `iio`), omijając abstrakcję PyADI-IIO. Przykładowo, możesz użyć właściwości `ctx` w PyADI-IIO, aby uzyskać dostęp do surowych buforów `libiio` i pobrać dane w formacie `int16`.

4. **Przykład dostępu do surowych danych `int16`**:
   Jeśli chcesz pominąć konwersję na `complex128` i uzyskać dane w formacie `int16`, możesz użyć następującego kodu:

   ```python
   import iio
   import adi

   # Połączenie z PlutoSDR
   sdr = adi.Pluto(uri="ip:192.168.2.1")
   ctx = sdr.ctx  # Uzyskaj kontekst libiio
   device = ctx.find_device("cf-adl9361-lpc")  # Znajdź urządzenie RX
   buffer = iio.Buffer(device, 1024, False)  # Stwórz bufor o rozmiarze 1024 próbek

   # Włącz kanały
   for channel in device.channels:
       if channel.input:
           channel.enabled = True

   # Pobierz dane
   buffer.refill()
   raw_data = buffer.read()  # Surowe dane w formacie int16
   print(raw_data)  # Tablica bajtów, wymaga ręcznego parsowania na I/Q
   ```

   W tym przypadku `raw_data` będzie tablicą bajtów reprezentującą próbki w formacie `int16`, które wymagają ręcznego rozdzielenia na składowe I i Q (np. co drugi element dla I i Q).

5. **Dlaczego PyADI-IIO nie zwraca `int16`?**:
   - **Uproszczenie dla użytkownika**: Konwersja na `complex128` eliminuje potrzebę ręcznego zarządzania formatem danych i skalowaniem, co jest szczególnie przydatne dla mniej zaawansowanych użytkowników.
   - **Uniwersalność**: Format `complex128` jest bardziej uniwersalny w ekosystemie Pythona, zwłaszcza w bibliotekach takich jak SciPy czy Matplotlib, które są często używane do analizy sygnałów.
   - **Unikanie błędów**: Ręczna obsługa `int16` wymaga zrozumienia szczegółów formatu danych (np. przesunięcia bitowego, endianness), co może prowadzić do błędów w aplikacjach.

6. **Jak uzyskać dane w `int16`, jeśli jest to potrzebne?**:
   - Użyj bezpośredniego dostępu do `libiio`, jak pokazano powyżej.
   - Alternatywnie, możesz skonwertować dane `complex128` z powrotem na `int16`, jeśli potrzebujesz tego formatu, np.:
     ```python
     import adi
     import numpy as np

     sdr = adi.Pluto(uri="ip:192.168.2.1")
     data = sdr.rx()  # Dane w complex128
     # Konwersja na int16 (skalowanie do zakresu int16)
     data_int16 = (data * 2048).astype(np.int16)  # Skalowanie do 12-bitowego zakresu
     print(data_int16)
     ```
     Uwaga: Skalowanie (np. mnożenie przez 2048) zależy od tego, jak chcesz mapować wartości zmiennoprzecinkowe na zakres `int16`.

**Podsumowanie**:
PyADI-IIO zwraca dane w formacie `complex128`, ponieważ konwertuje surowe próbki `int16` z ADC na format zmiennoprzecinkowy dla wygody i kompatybilności z Pythonem. Jeśli potrzebujesz surowych danych w formacie `int16`, musisz użyć niskopoziomowego API `libiio` lub ręcznie skonwertować dane z `complex128`. Dokumentacja PyADI-IIO i `libiio` (np.,) potwierdza, że surowy format danych w PlutoSDR to `int16`, ale PyADI-IIO upraszcza ich obsługę, stosując konwersję.[](https://wiki.analog.com/resources/tools-software/linux-software/libiio/iio_readdev)[](https://analogdevicesinc.github.io/pyadi-iio/libiio.html)