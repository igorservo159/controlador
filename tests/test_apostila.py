"""Validação cruzada com a apostila (§4.0 do CLAUDE.md)."""

import numpy as np

from lgr import (
    projeto_PD,
    projeto_PID_zigual,
    projeto_compensador_apb,
)


def test_apostila_PD():
    # §4.2 — G = 2/s², sd = -1 + 1.95j → z ≈ 2.41
    # A apostila usa G_norm = 1/s² (fatora o ganho K=2 da planta) e obtém
    # Kt = 2; depois faz Kc = Kt/K = 1.0. Aqui usamos a planta inteira e
    # Kc sai direto = 1.0.
    sd = -1 + 1.95j
    r = projeto_PD([2.0], [1.0, 0.0, 0.0], [1.0], [1.0], sd)
    assert abs(r["z"] - 2.41) < 0.02
    assert abs(r["Kc"] - 1.0) < 0.05


def test_apostila_PID_zigual():
    # §4.4 — G = 1/(s²+1), sd = -1 + sqrt(3)j → z ≈ 0.79, Kc ≈ 2.37
    sd = -1 + np.sqrt(3) * 1j
    r = projeto_PID_zigual([1.0], [1.0, 0.0, 1.0], [1.0], [1.0], sd)
    assert abs(r["z1"] - 0.79) < 0.02
    assert abs(r["Kc"] - 2.37) < 0.02


def test_apostila_lead():
    # §4.5 — G = 4/[s(s+2)], sd = -2 + 3.464j, z=3 fixo → p = 5.6
    # Aqui a apostila fixa o zero em z=3 e calcula o polo p.
    # Equivalente: planta efetiva G·(s+3)/[(s+2)·s] e usa a/(s+b) para achar b=p.
    # Implementação: incorporamos o zero fixo na "planta" passada ao a/(s+b).
    # G_eff(s)·H(s) = [4·(s+3)] / [s·(s+2)]
    sd = -2 + 3.464j
    r = projeto_compensador_apb(
        [4.0, 12.0],          # 4(s+3)
        [1.0, 2.0, 0.0],      # s(s+2)
        [1.0],
        [1.0],
        sd,
    )
    assert abs(r["b"] - 5.6) < 0.05
