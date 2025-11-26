# Inference with k-means

**Tytuł po polsku:** Wnioskowanie z k-średnich: nowe metody dla online balanced k-means

## Cel i postawienie problemu

Celem projektu opisanego w artykule było zaprojektowanie i empiryczna ocena metod wnioskowania (inferencji) opartych na algorytmie k-średnich, w szczególności dla wariantu **online balanced k-means**.

Rozważane zadanie polega na przewidywaniu ostatniej składowej wektora danych ($x_d$) na podstawie pozostałych składowych ($x_1, \ldots, x_{d-1}$) dla punktów pochodzących z rozkładów tworzących klastry. W literaturze brak jest jasnych wskazówek, jak prowadzić taką inferencję z wykorzystaniem k-means, co tworzy lukę między klasteryzacją a estymacją gęstości i predykcją.

## Dotychczasowe rozwiązania i tło

K-średnich to popularny, iteracyjny algorytm partycjonowania danych, przypisujący punkty do najbliższych centroidów i aktualizujący je do średnich przypisanych punktów. Wersje online przetwarzają dane napływające strumieniowo, a wariant **balanced** dąży do podobnych rozmiarów klastrów.

Nieparametryczna estymacja gęstości obejmuje histogramy, KDE, KNN i metody Woronoja, które oceniają gęstość bez zakładania konkretnej postaci rozkładu. Każda ma ograniczenia, estymacja Woronoja bywa mniej stronnicza względem lokalnej geometrii.

Mimo szerokiego stosowania k-means, jest luka badawcza – brak jest metodologii, która wykorzystywałaby strukturę klastrów do predykcji nieobserwowanych składowych punktu (inferencja).

## Nowe podejście i metodologia

Zaimplementowano **online balanced k-means**, łącząc aktualizacje online z mechanizmem równoważenia rozmiarów klastrów. Badano wpływ trzech hiperparametrów:

- liczby klastrów **k**
- współczynnika uczenia **α** (wpływ nowego punktu na centroid)
- czynnika równoważenia **β**

### Generowanie danych syntetycznych

Dane generowano syntetycznie w Pythonie, aby objąć:

- **Jeden klaster** (rozkłady: jednostajny, normalny, gamma; warianty: losowy, squared, cube – w dwóch ostatnich $x_d$ wyliczana z poprzednich składowych)
- **Dwa klastry** (mieszanki dwóch rozkładów w kolumnach)
- **Trzy klastry** (mieszanki: 3 × jednostajny, 3 × normalny, 2 × gamma + 1 × normalny)

**Wspólna idea:** cechy wejściowe pochodzą z rozkładów, a ostatnia składowa bywa funkcją wcześniejszych (co pozwala zweryfikować jakość predykcji).

### Metody wnioskowania

Opracowano siedem metod wnioskowania (przewidywania ostatniej składowej), które wykorzystują względną pozycję punktu wobec centroidów i/lub informację o klastrach:

1. **Metoda odległości euklidesowej** (najbliższy centroid)
2. **Metoda znormalizowanych wag** (wagi ~ estymaty prawdopodobieństw zależne od odległości)
3. **Metoda rozmiaru klastrów** (wagi z rozmiarów najbliższych klastrów)
4. **Średnia globalna + znormalizowane wagi** (łączenie)
5. **Znormalizowane wagi + rozmiar klastrów** (łączenie)
6. **Znormalizowane wagi + odległość euklidesowa** (łączenie)
7. **Rozmiar klastrów z wagami wykładniczymi** (rozszerzenie z parametrem beta)

Jako metrykę jakości wykorzystano **błąd kwadratowy predykcji** $x_d$ na zbiorze testowym.

## Najważniejsze wyniki

### Wpływ liczby klastrów k

Większa liczba klastrów **k** generalnie zmniejsza błędy wnioskowania. Dla badanego zakresu błędy stabilizują się przy k ≈ 100-1000, a **k ≈ 300** dawało najniższe błędy (w badanej konfiguracji).

### Liczba przypisań

Wzrost liczby przetworzonych punktów obniża straty uczenia, ale nie przekłada się wyraźnie na spadek błędów wnioskowania.

### Współczynnik uczenia α

Najlepsze wartości obserwowano przy **α ≈ 0,6** (następnie 0,4, 0,2). Duże α (blisko 1) dawało wyższe błędy.

### Czynnik równoważenia β

Optymalne wartości mieściły się w przybliżonym zakresie **0,21 do 0,7** (w zależności od danych).

### Porównanie metod

Połączenie znormalizowanych wag z rozmiarem klastrów oraz wariant z rozmiarami i wagami wykładniczymi dawały **wyższe błędy** niż pozostałe podejścia.

**Ogólny wniosek:** mimo poprawy dopasowania klastrów (spadku strat), błędy inferencji utrzymują się – badane metody nie zapewniły zadowalającej predykcji $x_d$ na rozważonych zbiorach.

## Wnioski

Wnioskowanie o brakującej/ostatniej składowej wyłącznie na podstawie geometrii k-means (centroidów i podziałów Woronoja) okazuje się trudne: **zmniejszenie strat klasteryzacji nie musi przekładać się na lepszą predykcję**.

## Podsumowanie

Projekt wypełnia lukę badawczą, proponując i testując pierwsze, systematyczne metody wnioskowania o brakującej składowej z użyciem k-średnich (wariant online balanced).

Dostarcza **negatywny, ale ważny wynik**: sama klasteryzacja nie wystarcza do wiarygodnej predykcji $x_d$ w rozważonych scenariuszach. Wskazane są metody hybrydowe i techniki inferencji lepiej modelujące relacje między cechami w danych wieloklastrowych.
