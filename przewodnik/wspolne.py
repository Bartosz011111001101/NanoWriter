"""Wspólne stałe i narzędzia materiałów przewodnika (bez rysowania)."""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from nanowriter.sample import generate

if TYPE_CHECKING:
    from nanowriter.model import GPT
    from nanowriter.tokenizer import BPETokenizer

ZDANIE: str = "Wieczorem stary kot zasnął przy piecu."
ROZMOWA_TEKST: str = (
    "A: Cześć, jak minął ci dzień?\n"
    "B: Spokojnie, trochę pracy i długi spacer.\n"
    "A: A ja cały dzień remontowałem kuchnię.\n"
    "B: To musiało być męczące.\n"
)
PROMPTY_ROZMOWY: tuple[str, ...] = (
    "cześć, co u ciebie?",
    "jestem dziś strasznie zmęczony",
    "co robiłeś w weekend?",
    "pada od rana, nie chce mi się wychodzić",
)


def _bajty_z_scalen(merges: list[tuple[int, int]], token_id: int) -> bytes:
    """Rekurencyjnie zwraca bajty tokenu na podstawie listy scaleń (bez sprawdzania zakresu).

    Przykład: _bajty_z_scalen([(97, 98)], 256) -> b"ab"
    """
    if token_id < 256:
        return bytes([token_id])
    pierwszy, drugi = merges[token_id - 256]
    return _bajty_z_scalen(merges, pierwszy) + _bajty_z_scalen(merges, drugi)


def bajty_tokenu(tokenizer: "BPETokenizer", token_id: int) -> bytes:
    """Zwraca bajty UTF-8 tokenu: id < 256 to jeden bajt, wyżej bajty obu składników scalenia.

    Przykład: tok = BPETokenizer([(97, 98), (256, 99)]); bajty_tokenu(tok, 97) -> b"a"
    Przykład: bajty_tokenu(tok, 256) -> b"ab"; bajty_tokenu(tok, 257) -> b"abc"
    Przykład: bajty_tokenu(tok, 258) -> ValueError (vocab_size == 258); id ujemne -> ValueError
    """
    if not 0 <= token_id < tokenizer.vocab_size:
        raise ValueError(f"id tokenu poza słownikiem: {token_id}")
    return _bajty_z_scalen(tokenizer.merges, token_id)


def _zapis_niedrukowalnego(znak: str) -> str:
    """Zwraca zapis znaku niedrukowalnego bez apostrofów, np. nowa linia -> \\n.

    Przykład: _zapis_niedrukowalnego("\\n") -> "\\\\n"
    """
    return znak.encode("unicode_escape").decode("ascii")


def etykieta_tokenu(tokenizer: "BPETokenizer", token_id: int) -> str:
    r"""Zwraca czytelny napis tokenu do wykresów (bez znaku zastępczego �).

    Bajty tokenu dekodowane jako UTF-8; bajty niepełnego znaku zapisane jako \xNN; znaki
    niedrukowalne (np. nowa linia) zapisane jako ich zapis (\n). Spacja zostaje spacją.
    Przykład: tok = BPETokenizer([(97, 98)]); etykieta_tokenu(tok, 256) -> "ab"
    Przykład: etykieta_tokenu(tok, 197) -> "\\xc5"  (4 znaki: pierwszy bajt litery ż)
    Przykład: etykieta_tokenu(tok, 10) -> "\\n"  (2 znaki: backslash i n)
    Przykład: etykieta_tokenu(BPETokenizer([(32, 107)]), 256) -> " k"
    """
    tekst = bajty_tokenu(tokenizer, token_id).decode("utf-8", errors="backslashreplace")
    return "".join(
        znak if znak.isprintable() else _zapis_niedrukowalnego(znak) for znak in tekst
    )


def zakoduj_zdanie(tokenizer: "BPETokenizer", zdanie: str, block_size: int) -> torch.Tensor:
    """Koduje zdanie do LongTensor[1, T] na CPU.

    Przykład: zakoduj_zdanie(BPETokenizer([]), "ab", 8) -> tensor([[97, 98]])
    Przykład: zakoduj_zdanie(BPETokenizer([(97, 98)]), "ab", 8) -> tensor([[256]])
    Przykład: zdanie dające T > block_size -> ValueError; zdanie "" -> ValueError
    """
    ids = tokenizer.encode(zdanie)
    if not ids:
        raise ValueError("zdanie koduje się do pustej listy tokenów")
    if len(ids) > block_size:
        raise ValueError(f"zdanie ma {len(ids)} tokenów, block_size to {block_size}")
    return torch.tensor([ids], dtype=torch.long)


def top_k_rozklad(
    tokenizer: "BPETokenizer", logity: torch.Tensor, k: int
) -> list[dict[str, Any]]:
    """Zwraca k najbardziej prawdopodobnych tokenów dla logitów jednej pozycji.

    logity: FloatTensor[V]. Prawdopodobieństwa to softmax(logity). Element listy:
    {"id": int, "token": etykieta_tokenu, "p": float}, malejąco po p; k > V obcięte do V.
    Przykład (p w przybliżeniu): tok = BPETokenizer([]); l = torch.zeros(256)
        l[97] = 2.0; l[98] = 1.0; top_k_rozklad(tok, l, 2)
        -> [{"id": 97, "token": "a", "p": 0.0280}, {"id": 98, "token": "b", "p": 0.0103}]
    Przykład: k < 1 albo logity.dim() != 1 -> ValueError
    """
    if logity.dim() != 1:
        raise ValueError(f"logity musi być jednowymiarowe, ma kształt {tuple(logity.shape)}")
    if k < 1:
        raise ValueError(f"k musi być co najmniej 1, jest {k}")
    prob = torch.softmax(logity.float(), dim=0)
    wartosci, indeksy = torch.topk(prob, min(k, prob.numel()))
    return [
        {"id": int(i), "token": etykieta_tokenu(tokenizer, int(i)), "p": float(p)}
        for p, i in zip(wartosci.tolist(), indeksy.tolist(), strict=True)
    ]


def entropia(prob: torch.Tensor) -> torch.Tensor:
    """Zwraca entropię Shannona (w nat) po ostatnim wymiarze rozkładu prob.

    Przykład: entropia(torch.tensor([0.5, 0.5])) -> tensor(0.6931) (ln 2)
    Przykład: entropia(torch.tensor([1.0, 0.0])) -> tensor(0.) (zero razy log zera to 0)
    Przykład: prob [3, 4] z wierszami jednostajnymi -> tensor([1.3863, 1.3863, 1.3863])
    """
    logarytm = torch.log(prob.clamp(min=1e-12))
    return torch.sum(-prob * logarytm, dim=-1)


def probka_tekstu(
    model: "GPT",
    tokenizer: "BPETokenizer",
    prompt: str,
    liczba_tokenow: int,
    *,
    device: str,
    seed: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """Zwraca prompt i dopisane liczba_tokenow tokenów jako tekst (generator z seedem).

    Przykład: dwa wywołania z tym samym modelem, promptem i seedem (CPU) -> ten sam tekst
    Przykład: tokenizer.encode(prompt) == [] albo liczba_tokenow < 1 -> ValueError
    """
    if liczba_tokenow < 1:
        raise ValueError(f"liczba_tokenow musi być co najmniej 1, jest {liczba_tokenow}")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt koduje się do pustej listy tokenów")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    wynik = generate(
        model,
        idx,
        liczba_tokenow,
        generator=generator,
        block_size=model.config.block_size,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(wynik[0].tolist())


def _na_json(dane: Any) -> Any:
    """Zamienia dane na typy zrozumiałe dla JSON: tensory i krotki na listy, Path na napis.

    Przykład: _na_json((1.0, float("nan"))) -> [1.0, None]
    """
    if isinstance(dane, torch.Tensor):
        return _na_json(dane.tolist())
    if isinstance(dane, dict):
        return {str(klucz): _na_json(wartosc) for klucz, wartosc in dane.items()}
    if isinstance(dane, list | tuple):
        return [_na_json(wartosc) for wartosc in dane]
    if isinstance(dane, Path):
        return str(dane)
    if isinstance(dane, float) and not math.isfinite(dane):
        return None
    return dane


def zapisz_json(dane: Any, sciezka: Path) -> Path:
    r"""Zapisuje dane jako JSON (UTF-8, wcięcie 2, "\n" na końcu) i zwraca sciezka.

    Tensory zamienia na listy, NaN i +-inf na null, Path na napis, krotki na listy;
    tworzy brakujące katalogi nadrzędne.
    Przykład: zapisz_json({"a": float("nan"), "b": torch.tensor([1.0, 2.0])}, p)
        -> plik: '{\n  "a": null,\n  "b": [\n    1.0,\n    2.0\n  ]\n}\n'
    """
    tekst = json.dumps(_na_json(dane), ensure_ascii=False, indent=2, allow_nan=False)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    sciezka.write_text(tekst + "\n", encoding="utf-8", newline="\n")
    return sciezka


def wczytaj_log(sciezka: Path) -> dict[str, list[float]]:
    """Wczytuje log treningu (JSON Lines) do list: kroki, train_loss, lr, val_kroki, val_loss.

    Przykład: linie {"step": 0, "train_loss": 6.9, "lr": 1e-05},
        {"step": 100, "train_loss": 5.2, "lr": 0.001}, {"step": 100, "val_loss": 5.5}
        -> {"kroki": [0, 100], "train_loss": [6.9, 5.2], "lr": [1e-05, 0.001],
            "val_kroki": [100], "val_loss": [5.5]}
    Przykład: pusty plik -> wszystkie pięć list puste; brak pliku -> FileNotFoundError
    """
    wynik: dict[str, list[float]] = {
        "kroki": [],
        "train_loss": [],
        "lr": [],
        "val_kroki": [],
        "val_loss": [],
    }
    tekst = sciezka.read_text(encoding="utf-8")
    for linia in tekst.splitlines():
        if not linia.strip():
            continue
        wpis = json.loads(linia)
        if "train_loss" in wpis:
            wynik["kroki"].append(wpis["step"])
            wynik["train_loss"].append(wpis["train_loss"])
            wynik["lr"].append(wpis["lr"])
        elif "val_loss" in wpis:
            wynik["val_kroki"].append(wpis["step"])
            wynik["val_loss"].append(wpis["val_loss"])
    return wynik
