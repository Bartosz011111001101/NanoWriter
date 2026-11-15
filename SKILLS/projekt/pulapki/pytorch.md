# Pułapki: PyTorch

Plik wymienny między projektami. Planner kopiuje pasujące wpisy do sekcji „Uwaga” w prompcie, a tester używa ich jako hipotez przy diagnozie. Nowy wpis dopisuje tester po rundzie poprawek, jeśli przyczyna może się powtórzyć w innym miejscu.

Format: **Obszar.** Objaw → przyczyna → jak uniknąć.

## Trening

- **Kroki optymalizatora.** Strata nie spada wcale → brak `opt.zero_grad()`, `opt.step()` przed `backward()`, LR = 0 albo parametry poza optymalizatorem → kolejność `zero_grad → forward → backward → step`, optymalizator dostaje `model.parameters()`.
- **NaN w stracie.** Strata zmienia się w `NaN` → za duży LR, maska z `-inf` w całym wierszu albo przepełnienie w fp16 → mniejszy LR, gradient clipping, bf16 zamiast fp16.
- **Przeuczenie na jednym batchu.** Model nie schodzi ze stratą na jednym batchu → złe przesunięcie `y`, zły kształt w `cross_entropy` albo zamrożone wagi → `y` to `x` przesunięte o jeden token, `cross_entropy(logits.view(-1, V), y.view(-1))`.
- **Strata startowa.** Strata po inicjalizacji dużo wyższa niż `ln(vocab_size)` → za duża inicjalizacja wag wyjściowych → małe odchylenie (np. 0.02) i zerowe biasy.
- **Tryb ewaluacji.** Strata walidacyjna skacze między wywołaniami albo rośnie zużycie pamięci → ewaluacja bez `model.eval()` i bez `torch.no_grad()` → oba, a po ewaluacji `model.train()`.

## Kształty i typy

- **dtype indeksów.** `IndexError` albo błąd typu w `nn.Embedding` → tokeny wczytane jako `uint16` → przed modelem `.long()`.
- **view i reshape.** `RuntimeError: view size is not compatible` → `view` po `transpose` na tensorze nieciągłym → `.contiguous().view(...)` albo `reshape`.
- **Wymiar softmaksu.** Wagi uwagi nie sumują się do 1 w wierszu → softmax po złym wymiarze → `softmax(..., dim=-1)` po wymiarze kluczy.
- **Broadcasting.** Kształt wyniku większy niż oczekiwany, bez błędu → cichy broadcasting `[B, T]` z `[T, B]` albo `[B]` z `[B, 1]` → sprawdzaj kształt każdego wyniku pośredniego z docstringu.

## Uwaga i generowanie

- **Maska przyczynowa.** Model „widzi przyszłość” albo nie widzi niczego → `triu` zamiast `tril` albo maska nałożona po softmaksie → `masked_fill(mask == 0, float("-inf"))` przed softmaksem, maska `tril`.
- **Przycięcie kontekstu.** Błąd kształtu przy długim generowaniu → wejście dłuższe niż `block_size` → przed `forward` bierz `idx[:, -block_size:]`.
- **Powtarzalność.** Wyniki różnią się między uruchomieniami przy tym samym seedzie → brak seeda albo dropout w trybie `train()` → seed przez `torch.Generator`, porównania w `eval()`.

## Urządzenia

- **Mieszane urządzenia.** `Expected all tensors to be on the same device` → tensor utworzony w `forward` bez `device=` → twórz go z `device=x.device` albo rejestruj jako bufor (`register_buffer`).
- **Checkpoint na inne urządzenie.** Wczytanie checkpointu z GPU na CPU się wysypuje → brak `map_location` → `torch.load(path, map_location=device)`.
