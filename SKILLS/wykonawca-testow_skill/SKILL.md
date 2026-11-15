---
name: wykonawca-testow_skill
description: Używaj, gdy dostajesz do wykonania specyfikację testów z Docs/Prompts/NN-zadanie/testy/ w projekcie NanoWriter.
---

# Wykonawca testów w NanoWriter

Dostajesz ścieżkę do jednej specyfikacji testów. Piszesz dokładnie te testy, które w niej są, w jednym pliku.

## Kroki

1. Przeczytaj specyfikację (pełna ścieżka od zlecającego), prompt kodu i kontrakt z jej nagłówka. `Docs/` jest w `.gitignore`, więc czytaj po pełnej ścieżce.
2. Napisz plik testów z nagłówka specyfikacji:
   - nazwy testów dokładnie takie jak w specyfikacji;
   - wejścia i oczekiwane wyniki dokładnie takie jak w specyfikacji;
   - import dokładnie taki jak w sekcji „Import”.
3. Sprawdź składnię: `uv run python -m py_compile <plik testów>`.
4. Uruchom: `uv run pytest <plik testów>`. Kod może jeszcze nie istnieć, bo powstaje równolegle. `ImportError` albo nieprzechodzące testy są wtedy w porządku. Zapisz wynik w raporcie.
5. Zapisz raport (szablon niżej).

## Zasady pisania testów

- Domyślnie CPU i malutki model (np. 2 warstwy, `d_model=32`, vocab 64, kontekst 16). Cały plik ma trwać kilka sekund.
- `torch.manual_seed(0)` na początku każdego testu z losowością.
- Model w testach porównujących wyjścia jest w `model.eval()`, bo dropout psuje porównania.
- Test wymagający GPU: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")`.
- Porównania tensorów: `torch.allclose(a, b, atol=1e-5)`, nie `==`.

## Przykład

Specyfikacja z testem maski przyczynowej daje taki kod:

```python
import torch
from nanowriter.model import GPT, GPTConfig

def test_causal_mask_hides_future():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (1, 16))
    changed = idx.clone()
    changed[0, 10] = (changed[0, 10] + 1) % config.vocab_size

    with torch.no_grad():
        a, b = model(idx), model(changed)

    # Pozycje przed zmianą nie mogą "widzieć" przyszłego tokenu.
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    # Od zmiany wzwyż wyjście musi się zmienić. Jeśli nie, model ignoruje wejście.
    assert not torch.allclose(a[:, 10:], b[:, 10:])
```

## Granice

- Zmieniasz tylko plik testów z nagłówka (i `tests/conftest.py`, jeśli nagłówek na to pozwala) oraz swój raport w `Docs/Postep/`.
- Nie zmieniasz niczego w `src/`, nawet gdy widzisz w kodzie błąd. Opisz go w raporcie.
- Nie dopisujesz testów spoza specyfikacji. Jeśli Twoim zdaniem czegoś brakuje, zapisz to w raporcie.
- Nie robisz commitów.

## Raport

Ścieżka: `Docs/Postep/NN-zadanie/testy/<nazwa pliku specyfikacji>`, np. `testy/01-pary.md` → `Postep/NN-zadanie/testy/01-pary.md`.

```markdown
# 01-pary (testy)

- **Wynik:** zrobione | zablokowane
- **Plik testów:** tests/test_tokenizer_01_pary.py
- **Liczba testów:** 3
- **Uruchomienie:** `uv run pytest tests/test_tokenizer_01_pary.py` → ImportError (kod jeszcze nie istnieje) | 3 passed | ...
- **Uwagi:** brak | <niejasność w specyfikacji, brakujący przypadek>
```

Odpowiedź dla zlecającego: ścieżka raportu i linia „Wynik”.
