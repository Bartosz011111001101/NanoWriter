# Konwencje projektu: NanoWriter

Czytają je planner, tester i wykonawcy na starcie.

## Kod

- Pełne adnotacje typów w każdej funkcji publicznej. Tensory adnotuj jako `torch.Tensor`, a kształt i dtype podawaj w docstringu, np. `idx: LongTensor[B, T]`.
- **Opis funkcji** (w skillach: „opis funkcji”) to w tym projekcie docstring po polsku: jedno zdanie, co robi, potem linia `Przykład: <wejście> -> <wyjście>` i przypadki brzegowe w tym samym formacie.
- **Wspólne przygotowanie testów** to fixture'y pytest w `tests/conftest.py`.
- Moduły w `src/` nie wypisują niczego (`print`). Postęp treningu raportuje tylko funkcja, która dostaje to jako argument, albo skrypt.
- Urządzenie (`device`) przychodzi jako argument albo z konfiguracji. Moduł nie wybiera sam CPU ani GPU.
- Losowość tylko przez przekazany `torch.Generator` albo seed z argumentu.
- Pliki tekstowe zawsze z `encoding="utf-8"`.
- Ścieżki jako `pathlib.Path`.
- Skrypty w `scripts/` tylko parsują argumenty i wywołują funkcje z modułów.

## Zależności i struktura

- Publiczne jest tylko to, co jest w kontrakcie modułu. Funkcja pomocnicza wykonawcy ma nazwę `_nazwa` i nikt spoza jej pliku jej nie importuje.
- Importy tylko z niższych warstw (kolejność w profilu).
- Na poziomie modułu tylko importy, definicje i stałe. Bez wczytywania plików, inicjalizacji CUDA i `print` przy imporcie.
- Bez globalnego stanu. Konfiguracja przychodzi argumentem (dataclass), a nie z globalnej zmiennej ani singletonu.
- Stała albo typ (np. `GPTConfig`) ma jedno miejsce definicji. Inne moduły go importują, a nie kopiują.
- Import potrzebny tylko do adnotacji idzie pod `if TYPE_CHECKING:`, żeby nie tworzył zależności w czasie działania.

## Testy

- Domyślnie CPU i malutki model (np. 2 warstwy, `d_model=32`, vocab 64, kontekst 16). Cały plik ma trwać kilka sekund.
- `torch.manual_seed(0)` na początku każdego testu z losowością.
- Model w testach porównujących wyjścia jest w `model.eval()`, bo dropout psuje porównania.
- Test wymagający GPU: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")`.
- Porównania tensorów: `torch.allclose(a, b, atol=1e-5)`, nie `==`.
- Pliki tymczasowe tylko przez fixture `tmp_path`.
