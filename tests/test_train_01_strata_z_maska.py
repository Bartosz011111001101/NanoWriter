"""Testy straty z maską: compute_loss, train_step, estimate_loss i train."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.model import GPT, GPTConfig
from nanowriter.train import (
    TrainConfig,
    compute_loss,
    estimate_loss,
    load_checkpoint,
    load_gpt,
    train,
    train_step,
)


def _gpt(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, d_model=16))


DATA = torch.randint(0, 16, (400,), generator=torch.Generator().manual_seed(0))
VAL = torch.randint(0, 16, (200,), generator=torch.Generator().manual_seed(1))
CFG = TrainConfig(
    batch_size=4,
    block_size=8,
    max_steps=20,
    learning_rate=1e-2,
    warmup_steps=4,
    lr_schedule="cosine",
    min_lr=1e-3,
    grad_clip=1.0,
    weight_decay=0.1,
    eval_every=5,
    eval_batches=2,
)


def _lt() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(0))
    targets = torch.randint(0, 5, (2, 3), generator=torch.Generator().manual_seed(1))
    return logits, targets


M = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.bool)

STRATY_BEZ_MASKI = [
    2.77981, 2.78233, 2.81744, 2.7664, 2.73115, 2.76006, 2.77137, 2.76621, 2.77309, 2.71733,
    2.68851, 2.8133, 2.75059, 2.73637, 2.77396, 2.71076, 2.8929, 2.70856, 2.68194, 2.77155,
]  # fmt: skip
EVAL_BEZ_MASKI = [(4, 2.80961), (9, 2.83127), (14, 2.84149), (19, 2.85064)]


def _zbieracz() -> tuple[list[tuple[int, float]], object]:
    zebrane: list[tuple[int, float]] = []

    def on_eval(step: int, val: float) -> None:
        zebrane.append((step, val))

    return zebrane, on_eval


def _batch_xym() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 16, (2, 8), generator=torch.Generator().manual_seed(10))
    y = torch.randint(0, 16, (2, 8), generator=torch.Generator().manual_seed(11))
    mask = torch.tensor(
        [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1]], dtype=torch.bool
    )
    return x, y, mask


def _opt(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)


def _logger(lista: list[tuple[int, float]]):
    def on_log(step: int, loss: float) -> None:
        lista.append((step, loss))

    return on_log


def _same_state(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------- compute_loss


def test_compute_loss_cztery_maski_z_opisu():
    logity = torch.zeros(1, 2, 4)
    logity[0, 1, 3] = 10.0
    t = torch.tensor([[0, 3]])

    assert compute_loss(logity, t, torch.tensor([[True, False]])).item() == pytest.approx(
        1.3863, abs=1e-4
    )
    assert compute_loss(logity, t, torch.tensor([[False, True]])).item() == pytest.approx(
        0.0001, abs=1e-4
    )
    assert compute_loss(logity, t, torch.tensor([[True, True]])).item() == pytest.approx(
        0.6932, abs=1e-4
    )
    assert compute_loss(logity, t, torch.tensor([[False, False]])).item() == 0.0
    assert compute_loss(logity, t).item() == pytest.approx(0.6932, abs=1e-4)


def test_compute_loss_bez_maski_ln4():
    loss = compute_loss(torch.zeros(1, 2, 4), torch.tensor([[0, 3]]))

    assert loss.item() == pytest.approx(math.log(4), abs=1e-4)
    assert loss.shape == ()
    assert loss.dtype == torch.float32


def test_compute_loss_srednia_po_pozycjach_z_maska_nie_po_wierszach():
    L, T = _lt()

    oracle = F.cross_entropy(L[M], T[M])
    srednia_wierszy = (
        F.cross_entropy(L[0][M[0]], T[0][M[0]]) + F.cross_entropy(L[1][M[1]], T[1][M[1]])
    ) / 2

    assert torch.allclose(compute_loss(L, T, M), oracle, atol=1e-6)
    assert abs(oracle.item() - srednia_wierszy.item()) > 1e-4


def test_compute_loss_typy_maski():
    L, T = _lt()
    oracle = F.cross_entropy(L[M], T[M])
    wynik = compute_loss(L, T, M)

    for maska in (M.to(torch.int64), M.to(torch.float32)):
        wynik_typu = compute_loss(L, T, maska)
        assert torch.allclose(wynik_typu, wynik, atol=1e-6)
        assert torch.allclose(wynik_typu, oracle, atol=1e-6)


def test_compute_loss_same_jedynki_rowna_sie_brak_maski():
    L, T = _lt()

    z_maska = compute_loss(L, T, torch.ones(2, 3, dtype=torch.bool))

    assert torch.allclose(z_maska, compute_loss(L, T), atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_compute_loss_maska_bez_jedynek(dtype):
    L, T = _lt()
    L2 = L.clone().requires_grad_(True)

    loss = compute_loss(L2, T, torch.zeros(2, 3, dtype=dtype))
    loss.backward()

    assert loss.item() == 0.0
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert L2.grad is not None
    assert L2.grad.shape == (2, 3, 5)
    assert not torch.isnan(L2.grad).any()
    assert (L2.grad == 0).all()


def test_compute_loss_cele_na_pozycjach_z_maska_0_nie_wplywaja():
    L, T = _lt()
    T2 = T.clone()
    T2[~M] = (T + 1)[~M] % 5
    L_a = L.clone().requires_grad_(True)
    L_b = L.clone().requires_grad_(True)

    loss_a = compute_loss(L_a, T, M)
    loss_b = compute_loss(L_b, T2, M)
    loss_a.backward()
    loss_b.backward()

    assert not torch.equal(T, T2)
    assert torch.allclose(loss_a, loss_b, atol=1e-7)
    assert torch.allclose(L_a.grad, L_b.grad, atol=1e-7)
    assert (L_a.grad[~M] == 0).all()
    assert (L_a.grad[M].abs().sum(dim=-1) > 0).all()


def test_compute_loss_bledy():
    with pytest.raises(ValueError):
        compute_loss(
            torch.zeros(1, 2, 4), torch.tensor([[0, 1]]), torch.ones(1, 3, dtype=torch.bool)
        )
    L, T = _lt()
    with pytest.raises(ValueError):
        compute_loss(L, T, torch.ones(3, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        compute_loss(torch.zeros(1, 2, 4), torch.tensor([[0, 1, 2]]))
    with pytest.raises(ValueError):
        compute_loss(torch.zeros(2, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        compute_loss(torch.zeros(2, 4), torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool))


# ------------------------------------------------------------------ train_step


def test_train_step_zwraca_strate_z_maska_sprzed_aktualizacji():
    model = _gpt()
    x, y, mask = _batch_xym()
    przed = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        expected = compute_loss(model(x), y, mask).item()

    loss = train_step(model, _opt(model), x, y, mask=mask)

    assert loss == pytest.approx(expected, abs=1e-6)
    po = model.state_dict()
    assert any(not torch.equal(przed[k], po[k]) for k in przed)


def test_train_step_bez_maski_jak_dotad():
    model_a, model_b, model_ref = _gpt(), _gpt(), _gpt()
    x, y, _ = _batch_xym()
    with torch.no_grad():
        expected = compute_loss(model_ref(x), y).item()

    loss_a = train_step(model_a, _opt(model_a), x, y)
    loss_b = train_step(model_b, _opt(model_b), x, y, mask=None)

    assert loss_a == pytest.approx(loss_b, abs=1e-9)
    assert loss_a == pytest.approx(expected, abs=1e-6)
    assert loss_b == pytest.approx(expected, abs=1e-6)
    assert _same_state(model_a, model_b)


def test_train_step_maska_bez_jedynek_nie_zmienia_wag():
    model = _gpt()
    x, y, _ = _batch_xym()
    przed = {k: v.clone() for k, v in model.state_dict().items()}

    loss = train_step(model, _opt(model), x, y, mask=torch.zeros(2, 8, dtype=torch.bool))

    assert loss == 0.0
    for k, v in model.state_dict().items():
        assert torch.equal(v, przed[k]), k
        assert torch.isfinite(v).all(), k


def test_train_step_cele_z_maska_0_nie_wplywaja_na_wagi():
    model_a, model_b = _gpt(), _gpt()
    x, y, mask = _batch_xym()
    y2 = y.clone()
    y2[~mask] = ((y + 3) % 16)[~mask]

    loss_a = train_step(model_a, _opt(model_a), x, y, mask=mask)
    loss_b = train_step(model_b, _opt(model_b), x, y2, mask=mask)

    assert not torch.equal(y, y2)
    assert loss_a == pytest.approx(loss_b, abs=1e-7)
    sa, sb = model_a.state_dict(), model_b.state_dict()
    for k in sa:
        assert torch.allclose(sa[k], sb[k], atol=1e-7), k


def test_train_step_z_grad_clip_i_maska_skonczone():
    model = _gpt()
    x, y, mask = _batch_xym()
    with torch.no_grad():
        expected = compute_loss(model(x), y, mask).item()

    loss = train_step(model, _opt(model), x, y, grad_clip=1.0, mask=mask)

    assert math.isfinite(loss)
    assert loss == pytest.approx(expected, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_train_step_cuda_bf16_z_maska():
    model = _gpt().to("cuda")
    x, y, mask = _batch_xym()

    loss = train_step(
        model, _opt(model), x.to("cuda"), y.to("cuda"), bf16=True, mask=mask.to("cuda")
    )

    assert isinstance(loss, float)
    assert math.isfinite(loss)


# --------------------------------------------------------------- estimate_loss


def _emb() -> nn.Embedding:
    torch.manual_seed(0)
    return nn.Embedding(8, 8)


E = torch.randint(0, 8, (41,), generator=torch.Generator().manual_seed(3))
EM = torch.arange(41) % 3 == 0


def _oracle(
    emb: nn.Embedding, data: torch.Tensor, mask: torch.Tensor, n_windows: int, block: int = 4
) -> float:
    """Średnia entropia po wszystkich pozycjach z maską 1 w oknach 0..n_windows-1."""
    logits, targets = [], []
    with torch.no_grad():
        for i in range(n_windows):
            x_i = data[block * i : block * i + block]
            y_i = data[block * i + 1 : block * i + block + 1]
            m_i = mask[block * i + 1 : block * i + block + 1].bool()
            logits.append(emb.weight[x_i][m_i])
            targets.append(y_i[m_i])
        return F.cross_entropy(torch.cat(logits), torch.cat(targets)).item()


def test_estimate_loss_maska_ln4_przyklad():
    model = nn.Embedding(4, 4)
    with torch.no_grad():
        model.weight.zero_()
    data = torch.arange(9) % 4

    wynik_float = estimate_loss(model, data, 2, 4, "cpu", mask=torch.ones(9))
    wynik_bool = estimate_loss(model, data, 2, 4, "cpu", mask=torch.ones(9, dtype=torch.bool))

    assert wynik_float == pytest.approx(1.3862943, abs=1e-5)
    assert wynik_bool == pytest.approx(1.3862943, abs=1e-5)
    with pytest.raises(ValueError):
        estimate_loss(model, data, 2, 4, "cpu", mask=torch.zeros(9))
    with pytest.raises(ValueError):
        estimate_loss(model, data, 2, 4, "cpu", mask=torch.ones(8))


def test_estimate_loss_same_jedynki_rowna_sie_brak_maski():
    model = _gpt()
    model.eval()

    for max_batches in (None, 3):
        bez = estimate_loss(model, DATA, 4, 8, "cpu", max_batches=max_batches)
        for maska in (torch.ones(400, dtype=torch.bool), torch.ones(400, dtype=torch.int64)):
            z = estimate_loss(model, DATA, 4, 8, "cpu", max_batches=max_batches, mask=maska)
            assert z == pytest.approx(bez, abs=1e-6)


def test_estimate_loss_srednia_tylko_z_pozycji_z_maska():
    emb = _emb()
    oracle = _oracle(emb, E, EM, 10)

    wynik = estimate_loss(_emb(), E, 3, 4, "cpu", mask=EM)
    wynik_int = estimate_loss(_emb(), E, 3, 4, "cpu", mask=EM.to(torch.int64))
    bez_maski = estimate_loss(_emb(), E, 3, 4, "cpu")

    assert wynik == pytest.approx(oracle, abs=1e-6)
    assert wynik_int == pytest.approx(oracle, abs=1e-6)
    assert abs(wynik - bez_maski) > 1e-4


def test_estimate_loss_max_batches_z_maska():
    emb = _emb()

    dwa = estimate_loss(_emb(), E, 3, 4, "cpu", max_batches=2, mask=EM)
    jeden = estimate_loss(_emb(), E, 3, 4, "cpu", max_batches=1, mask=EM)

    assert dwa == pytest.approx(_oracle(emb, E, EM, 6), abs=1e-6)
    assert jeden == pytest.approx(_oracle(emb, E, EM, 3), abs=1e-6)


def test_estimate_loss_przesuniecie_maski_o_jeden():
    data = torch.tensor([3, 1, 4, 1, 5, 2, 6, 5, 3])
    mask = torch.zeros(9, dtype=torch.bool)
    mask[5] = True
    model = _emb()

    wynik = estimate_loss(model, data, 2, 4, "cpu", mask=mask)

    oczekiwane = F.cross_entropy(model.weight[5].unsqueeze(0), torch.tensor([2])).item()
    assert wynik == pytest.approx(oczekiwane, abs=1e-6)

    mask0 = torch.zeros(9, dtype=torch.bool)
    mask0[0] = True
    with pytest.raises(ValueError):
        estimate_loss(model, data, 2, 4, "cpu", mask=mask0)


def test_estimate_loss_batche_bez_jedynek():
    mask = torch.arange(41) >= 30
    emb = _emb()

    wynik = estimate_loss(_emb(), E, 3, 4, "cpu", mask=mask)
    assert wynik == pytest.approx(_oracle(emb, E, mask, 10), abs=1e-6)

    for max_batches in (1, 2):
        with pytest.raises(ValueError):
            estimate_loss(_emb(), E, 3, 4, "cpu", max_batches=max_batches, mask=mask)
    trzy = estimate_loss(_emb(), E, 3, 4, "cpu", max_batches=3, mask=mask)
    assert trzy == pytest.approx(_oracle(emb, E, mask, 9), abs=1e-6)


def test_estimate_loss_niezalezne_od_batch_size():
    wyniki = [estimate_loss(_emb(), E, bs, 4, "cpu", mask=EM) for bs in (1, 3, 10)]

    assert wyniki[1] == pytest.approx(wyniki[0], abs=1e-6)
    assert wyniki[2] == pytest.approx(wyniki[0], abs=1e-6)


def test_estimate_loss_bledy_maski():
    with pytest.raises(ValueError):
        estimate_loss(_emb(), E, 3, 4, "cpu", mask=torch.ones(40, dtype=torch.bool))
    with pytest.raises(ValueError):
        estimate_loss(_emb(), E, 3, 4, "cpu", mask=torch.ones(41, 1, dtype=torch.bool))


def test_estimate_loss_z_maska_przywraca_tryb_i_nie_zostawia_gradientow():
    model = _gpt()
    maska = DATA % 2 == 0

    model.train()
    estimate_loss(model, DATA, 4, 8, "cpu", mask=maska)
    assert model.training is True

    model.eval()
    estimate_loss(model, DATA, 4, 8, "cpu", mask=maska)
    assert model.training is False

    assert all(p.grad is None for p in model.parameters())
    assert torch.is_grad_enabled() is True


# ----------------------------------------------------------------------- train


def test_train_bez_maski_te_same_straty_co_przed_zmiana():
    for kwargs in ({}, {"mask": None, "val_mask": None}):
        zebrane, on_eval = _zbieracz()

        losses = train(_gpt(), DATA, CFG, "cpu", val_data=VAL, on_eval=on_eval, **kwargs)

        assert len(losses) == 20
        assert losses == pytest.approx(STRATY_BEZ_MASKI, abs=1e-4)
        assert [s for s, _ in zebrane] == [s for s, _ in EVAL_BEZ_MASKI]
        assert [v for _, v in zebrane] == pytest.approx([v for _, v in EVAL_BEZ_MASKI], abs=1e-4)


def test_train_maska_same_jedynki_jak_bez_maski():
    zebrane, on_eval = _zbieracz()

    losses = train(
        _gpt(),
        DATA,
        CFG,
        "cpu",
        val_data=VAL,
        on_eval=on_eval,
        mask=torch.ones(400, dtype=torch.bool),
        val_mask=torch.ones(200, dtype=torch.bool),
    )

    assert len(losses) == 20
    for wynik, oczekiwana in zip(losses, STRATY_BEZ_MASKI, strict=True):
        assert wynik == pytest.approx(oczekiwana, abs=1e-4)
    assert [s for s, _ in zebrane] == [s for s, _ in EVAL_BEZ_MASKI]
    for (_, wynik), (_, oczekiwana) in zip(zebrane, EVAL_BEZ_MASKI, strict=True):
        assert wynik == pytest.approx(oczekiwana, abs=1e-4)


def test_train_maska_spadek_straty_na_powtarzalnych_danych():
    data = torch.arange(1000) % 8
    mask = data >= 4
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=2, d_model=32))
    cfg = TrainConfig(batch_size=8, block_size=16, max_steps=100, learning_rate=1e-2)

    losses = train(model, data, cfg, "cpu", mask=mask)

    assert len(losses) == 100
    assert all(math.isfinite(v) for v in losses)
    assert sum(losses[:10]) / 10 > 0.5
    assert sum(losses[-10:]) / 10 < 0.1


def test_train_maska_uczy_sie_tylko_przewidywalnych_celow():
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 8, (2001,), generator=g)
    data[1::2] = data[0:-1:2]
    mask = torch.zeros(2001, dtype=torch.bool)
    mask[1::2] = True
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=2, d_model=32))
    cfg = TrainConfig(batch_size=16, block_size=16, max_steps=300, learning_rate=1e-2)

    losses = train(model, data, cfg, "cpu", mask=mask)

    assert sum(losses[-30:]) / 30 < 0.05
    assert estimate_loss(model, data, 16, 16, "cpu", mask=mask) < 0.05
    assert estimate_loss(model, data, 16, 16, "cpu") > 2.0


def test_train_wznowienie_z_maska_bramka(tmp_path):
    cfg = dataclasses.replace(CFG, max_steps=12, eval_every=4)
    mask = torch.arange(400) % 3 != 0
    val_mask = torch.arange(200) % 2 == 0
    p = tmp_path / "ck.pt"

    zebrane_a, on_eval_a = _zbieracz()
    model_a = _gpt()
    losses_a = train(
        model_a, DATA, cfg, "cpu", val_data=VAL, mask=mask, val_mask=val_mask, on_eval=on_eval_a
    )

    zebrane_b, on_eval_b = _zbieracz()
    losses_b1 = train(
        _gpt(),
        DATA,
        cfg,
        "cpu",
        val_data=VAL,
        mask=mask,
        val_mask=val_mask,
        on_eval=on_eval_b,
        until_step=5,
        checkpoint_path=p,
    )
    model_b2 = _gpt(1)
    losses_b2 = train(
        model_b2,
        DATA,
        cfg,
        "cpu",
        val_data=VAL,
        mask=mask,
        val_mask=val_mask,
        on_eval=on_eval_b,
        resume_from=p,
    )

    assert len(losses_b1) == 5
    assert len(losses_b2) == 7
    assert losses_b1 + losses_b2 == losses_a
    assert zebrane_b == zebrane_a
    sa, sb = model_a.state_dict(), model_b2.state_dict()
    for k in sa:
        assert torch.allclose(sa[k], sb[k], atol=1e-6), k


def test_train_on_eval_dostaje_strate_z_maska():
    cfg = dataclasses.replace(CFG, max_steps=6, eval_every=6, eval_batches=2)
    val_mask = torch.arange(200) % 4 == 0
    model = _gpt()
    zebrane, on_eval = _zbieracz()

    train(
        model,
        DATA,
        cfg,
        "cpu",
        val_data=VAL,
        val_mask=val_mask,
        on_eval=on_eval,
        mask=torch.ones(400, dtype=torch.bool),
    )

    assert len(zebrane) == 1
    assert zebrane[0][0] == 5
    z_maska = estimate_loss(model, VAL, 4, 8, "cpu", max_batches=2, mask=val_mask)
    bez_maski = estimate_loss(model, VAL, 4, 8, "cpu", max_batches=2)
    assert z_maska == pytest.approx(zebrane[0][1], abs=1e-6)
    assert abs(bez_maski - zebrane[0][1]) > 1e-4


def test_train_bledy_masek_przed_pierwszym_krokiem(tmp_path):
    przypadki = [
        {"mask": torch.ones(399, dtype=torch.bool)},
        {"val_mask": torch.ones(200, dtype=torch.bool)},
        {"val_data": VAL, "val_mask": torch.ones(199, dtype=torch.bool)},
    ]
    cfg = dataclasses.replace(CFG, max_steps=5, log_every=1)

    for kwargs in przypadki:
        model = _gpt()
        przed = {k: v.clone() for k, v in model.state_dict().items()}
        wywolania: list[tuple[int, float]] = []
        sciezka = tmp_path / "x.pt"

        with pytest.raises(ValueError):
            train(
                model,
                DATA,
                cfg,
                "cpu",
                _logger(wywolania),
                checkpoint_path=sciezka,
                **kwargs,
            )

        assert wywolania == []
        assert not sciezka.exists()
        for k, v in model.state_dict().items():
            assert torch.equal(v, przed[k]), k


def test_train_douczanie_z_checkpointu_z_maska(tmp_path):
    p = tmp_path / "wstepny.pt"
    q = tmp_path / "rozmowy.pt"
    train(
        _gpt(),
        DATA,
        dataclasses.replace(CFG, max_steps=3, eval_every=0),
        "cpu",
        checkpoint_path=p,
    )
    assert load_checkpoint(p)["step"] == 3
    bajty = p.read_bytes()

    model2 = load_gpt(p, "cpu")
    losses = train(
        model2,
        DATA,
        dataclasses.replace(CFG, max_steps=6, eval_every=0),
        "cpu",
        mask=torch.arange(400) % 2 == 0,
        checkpoint_path=q,
    )

    assert load_checkpoint(q)["step"] == 6
    assert p.read_bytes() == bajty
    assert len(losses) == 6
    assert all(math.isfinite(v) for v in losses)
    assert load_checkpoint(q)["model_config"] == load_checkpoint(p)["model_config"]

    model3 = load_gpt(q, "cpu")
    idx = torch.randint(0, 16, (1, 8), generator=torch.Generator().manual_seed(5))
    model2.eval()
    with torch.no_grad():
        assert torch.allclose(model3(idx), model2(idx), atol=1e-5)
