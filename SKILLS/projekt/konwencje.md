# Konwencje projektu: NanoWriter

Czytają je wykonawcy kodu i testów na starcie. Planner nie powtarza ich w promptach.

## Kod

- Pełne adnotacje typów w każdej funkcji publicznej. Tensory adnotuj jako `torch.Tensor`, a kształt i dtype podawaj w docstringu, np. `idx: LongTensor[B, T]`.
- Docstringi po polsku, w formacie ze szkieletu promptu.
- Moduły w `src/` nie wypisują niczego (`print`). Postęp treningu raportuje tylko funkcja, która dostaje to jako argument, albo skrypt.
- Urządzenie (`device`) przychodzi jako argument albo z konfiguracji. Moduł nie wybiera sam CPU ani GPU.
- Losowość tylko przez przekazany `torch.Generator` albo seed z argumentu.
- Pliki tekstowe zawsze z `encoding="utf-8"`.
- Ścieżki jako `pathlib.Path`.
- Skrypty w `scripts/` tylko parsują argumenty i wywołują funkcje z modułów.

## Testy

- Domyślnie CPU i malutki model (np. 2 warstwy, `d_model=32`, vocab 64, kontekst 16). Cały plik ma trwać kilka sekund.
- `torch.manual_seed(0)` na początku każdego testu z losowością.
- Model w testach porównujących wyjścia jest w `model.eval()`, bo dropout psuje porównania.
- Test wymagający GPU: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")`.
- Porównania tensorów: `torch.allclose(a, b, atol=1e-5)`, nie `==`.
- Pliki tymczasowe tylko przez fixture `tmp_path`.
