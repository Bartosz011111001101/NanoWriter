# Katalog testów: NanoWriter

Testy obowiązkowe dla tej dziedziny. Tester sprawdza, czy każdy pasujący wpis z katalogu ma test w specyfikacji.

## Przypadki brzegowe

- Tekst: pusty `""`, polskie znaki `"Zażółć gęślą jaźń"`, emoji, białe znaki `"\n\t"`.
- Tensory: `B=1` i `T=1`, a także `T == block_size`.
- Kształt i dtype każdego zwracanego tensora.

## Testy według modułu

| Moduł | Test | Co sprawdza |
| --- | --- | --- |
| `tokenizer` | roundtrip | `decode(encode(x)) == x` dla wszystkich przypadków brzegowych tekstu |
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
| `train` | wznowienie | trening N kroków = N/2 + wznowienie + N/2 |
| `sample` | długość | `generate` zwraca dokładnie zadaną liczbę nowych tokenów |
| `sample` | powtarzalność | ten sam seed daje ten sam tekst |
| `sample` | KV cache | wynik z cache i bez cache jest identyczny |

## Przykład: test maski przyczynowej

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
