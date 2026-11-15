---
name: tester_skill
description: Używaj, gdy dla kodu NanoWriter trzeba napisać testy pytest według kontraktu, sprawdzić, czy test nie przechodzi z właściwego powodu, albo zdiagnozować typowy błąd modelu (strata nie spada, NaN, wyciek przyszłości przez maskę).
---

# Testowanie w NanoWriter

## Zasady

- Testy wynikają z kontraktu: każdy niezmiennik z sekcji „Niezmienniki” ma swój test.
- Domyślnie testy idą na CPU, na malutkim modelu (np. 2 warstwy, `d_model=32`, vocab 64, kontekst 16). Cały plik testów wykonuje się w kilka sekund.
- Seed na początku każdego testu z losowością: `torch.manual_seed(0)`.
- Model w testach porównujących wyjścia jest w `model.eval()`, bo dropout psuje porównania.
- Testy wymagające GPU: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")`.
- Wspólne fixture'y (malutka konfiguracja, model, tokenizer) leżą w `tests/conftest.py`.

## Właściwy powód porażki (RED)

| Właściwy powód | Niewłaściwy powód, czyli test do poprawy |
| --- | --- |
| Brak modułu, klasy albo funkcji, którą zadanie ma dodać | Literówka w nazwie z kontraktu |
| Funkcja istnieje, ale zwraca zły wynik | Błąd w fixture albo w danych testowych |
| `NotImplementedError` w zaślepce | Zły import istniejącego modułu |

## Katalog testów

| Moduł | Test | Co sprawdza |
| --- | --- | --- |
| `tokenizer` | roundtrip | `decode(encode(x)) == x` dla `""`, `"Zażółć gęślą jaźń"`, emoji, `"\n\t"` |
| `tokenizer` | zakres id | każde id spełnia `0 <= id < vocab_size` |
| `tokenizer` | save/load | po `load` wynik `encode` jest identyczny |
| `dataset` | format `.bin` | dtype i długość zgodne z kontraktem |
| `dataset` | przesunięcie | `y[:, :-1] == x[:, 1:]` |
| `model` | kształty | `logits.shape == (B, T, vocab_size)` |
| `model` | maska przyczynowa | zmiana tokenu na pozycji `t` nie zmienia wyjść na pozycjach `< t` |
| `model` | strata startowa | po inicjalizacji strata ≈ `ln(vocab_size)` (±10%) |
| `model` | gradienty | po `backward` każdy parametr ma niezerowy `grad` |
| `model` | przeuczenie | na jednym batchu strata spada poniżej ~0,1 w kilkuset krokach |
| `train` | checkpoint | po zapisie i odczycie model daje identyczne `logits` |
| `train` | wznowienie | trening N kroków = trening N/2 kroków + wznowienie + N/2 kroków |
| `sample` | długość | `generate` zwraca dokładnie zadaną liczbę nowych tokenów |
| `sample` | powtarzalność | ten sam seed daje ten sam tekst |
| `sample` | KV cache | wynik z cache i bez cache jest identyczny |

## Przykład: testy modelu

Nazwy (`GPTConfig`, `GPT`) dopasuj do kontraktu `Docs/Kontrakty/model.md`.

```python
# tests/conftest.py
import pytest
import torch
from nanowriter.model import GPT, GPTConfig

@pytest.fixture
def tiny_config():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)

@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return GPT(tiny_config).eval()
```

```python
# tests/test_model.py
import math
import torch
import torch.nn.functional as F

def test_causal_mask_hides_future(tiny_model, tiny_config):
    torch.manual_seed(0)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 16))
    changed = idx.clone()
    changed[0, 10] = (changed[0, 10] + 1) % tiny_config.vocab_size

    with torch.no_grad():
        a, b = tiny_model(idx), tiny_model(changed)

    # Pozycje przed zmianą nie mogą "widzieć" przyszłego tokenu.
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    # Pozycje od zmiany wzwyż powinny się zmienić. Jeśli nie, model ignoruje wejście.
    assert not torch.allclose(a[:, 10:], b[:, 10:])

def test_initial_loss_near_uniform(tiny_model, tiny_config):
    torch.manual_seed(0)
    x = torch.randint(0, tiny_config.vocab_size, (4, 16))
    y = torch.randint(0, tiny_config.vocab_size, (4, 16))
    with torch.no_grad():
        logits = tiny_model(x)
    loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), y.view(-1))
    # Świeży model powinien zgadywać mniej więcej równomiernie.
    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 0.1 * math.log(tiny_config.vocab_size)

def test_overfits_single_batch(tiny_model, tiny_config):
    torch.manual_seed(0)
    tiny_model.train()
    x = torch.randint(0, tiny_config.vocab_size, (4, 16))
    y = torch.roll(x, shifts=-1, dims=1)
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)
    for _ in range(300):
        logits = tiny_model(x)
        loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Jeśli model nie potrafi zapamiętać jednego batcha, w kodzie jest błąd.
    assert loss.item() < 0.1
```

## Diagnoza typowych porażek

Podaj ją w raporcie jako hipotezę. Kodu w `src/` nie poprawiasz.

| Objaw | Najczęstsza przyczyna |
| --- | --- |
| Strata nie spada wcale | brak `opt.zero_grad()`, `opt.step()` przed `backward`, LR = 0, parametry poza optymalizatorem |
| Strata `NaN` | za duży LR, maska z `-inf` w całym wierszu, przepełnienie w fp16 (użyj bf16) |
| Test maski przyczynowej nie przechodzi | maska `triu` zamiast `tril` albo maska nałożona po softmaksie |
| Strata startowa dużo wyższa niż `ln(V)` | za duża inicjalizacja wag wyjściowych |
| Nie przeucza się na jednym batchu | błąd w przesunięciu `y`, zły kształt w `cross_entropy`, zamrożone wagi |
| Roundtrip psuje polskie znaki | tokenizer działa na znakach zamiast na bajtach UTF-8 albo dekoduje token po tokenie |
| Wyniki różnią się między uruchomieniami | brak seeda, `model.train()` z dropoutem w teście porównującym |
