"""Testes dos 6 exercícios da lista DCA-3701 (§4.1–§4.8 do CLAUDE.md)."""

import numpy as np

from lgr import (
    c2d_euler,
    c2d_tustin,
    malha_fechada,
    polo_desejado,
    projeto_PD,
    projeto_PI,
    projeto_PID_zigual,
    projeto_compensador_apb,
    tf_mul,
    wn_min,
    zeta_de_mp,
)


def _polo_mais_proximo(polos, sd):
    return float(np.min(np.abs(polos - sd)))


# ---------------------------------------------------------------------------
# Ex 1 — PD (Mp ≤ 10%, ts(5%) < 4s)
# ---------------------------------------------------------------------------

def test_ex1_PD():
    zeta = zeta_de_mp(10.0)
    wn = wn_min(zeta, 4.0, "5%")
    sd = polo_desejado(zeta, wn)

    numG = [4.0, 16.0]
    denG = [1.0, 4.0, 4.0, 0.0]
    numH = [1.0]
    denH = [1.0]

    r = projeto_PD(numG, denG, numH, denH, sd)
    assert abs(r["z"] - 8.6598) < 1e-3
    assert abs(r["Kc"] - 0.030458) < 1e-5

    nT, dT = malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
    assert _polo_mais_proximo(np.roots(dT), sd) < 1e-9


# ---------------------------------------------------------------------------
# Ex 2 — PD com planta instável (ζ=0.7, ωn=0.5)
# ---------------------------------------------------------------------------

def test_ex2_PD():
    sd = polo_desejado(0.7, 0.5)

    numG = [1.0]
    denG = [10000.0, 0.0, -11772.0]
    numH = [1.0]
    denH = [1.0]

    r = projeto_PD(numG, denG, numH, denH, sd)
    assert abs(r["z"] - 2.0389) < 1e-3
    assert abs(r["Kc"] - 7000.0) < 1.0

    nT, dT = malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
    assert _polo_mais_proximo(np.roots(dT), sd) < 1e-9


# ---------------------------------------------------------------------------
# Ex 3 — PI (sd = -4 ± 4j)
# ---------------------------------------------------------------------------

def test_ex3_PI():
    sd = -4 + 4j

    numG = [5.0, 25.0, 20.0]
    denG = [1.0, 4.0, 4.0]
    numH = [0.2]
    denH = [1.0, 1.0]

    r = projeto_PI(numG, denG, numH, denH, sd)
    assert abs(r["z"] - 24.0 / 7.0) < 1e-3
    assert abs(r["Kc"] - 7.0) < 1e-3
    assert abs(r["Ki"] - 24.0) < 1e-3

    nT, dT = malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
    assert _polo_mais_proximo(np.roots(dT), sd) < 1e-9


# ---------------------------------------------------------------------------
# Ex 4 — PID z₁ = z₂ (Mp ≤ 20%, ts(2%) < 5s)
# ---------------------------------------------------------------------------

def test_ex4_PID():
    zeta = zeta_de_mp(20.0)
    wn = wn_min(zeta, 5.0, "2%")
    sd = polo_desejado(zeta, wn)

    numG = [5.0]
    denG = [1.0, 12.0, 22.0, 20.0]
    numH = [0.4]
    denH = [1.0]

    r = projeto_PID_zigual(numG, denG, numH, denH, sd)
    assert abs(r["z1"] - 2.0490) < 1e-3
    assert abs(r["Kc"] - 3.1360) < 1e-3

    nT, dT = malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
    assert _polo_mais_proximo(np.roots(dT), sd) < 1e-9


# ---------------------------------------------------------------------------
# Ex 5(a) — PID z₁ = z₂ (Mp ≤ 10%, ts(5%) < 3s)
# ---------------------------------------------------------------------------

def test_ex5a_PID():
    zeta = zeta_de_mp(10.0)
    wn = wn_min(zeta, 3.0, "5%")
    sd = polo_desejado(zeta, wn)

    numG = [5.0, 15.0]
    denG = [1.0, 4.0, 0.0]
    numH = [1.0]
    denH = [1.0, 1.0]

    r = projeto_PID_zigual(numG, denG, numH, denH, sd)
    assert abs(r["z1"] - 1.3322) < 1e-3
    assert abs(r["Kc"] - 0.5390) < 1e-3

    nT, dT = malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
    assert _polo_mais_proximo(np.roots(dT), sd) < 1e-9


# ---------------------------------------------------------------------------
# Ex 5(b) — Tustin com T = 2 s
# ---------------------------------------------------------------------------

def test_ex5b_tustin():
    Kc, z = 0.5390, 1.3322
    num_Gc = Kc * np.array([1.0, 2.0 * z, z * z])
    den_Gc = np.array([1.0, 0.0])

    nz, dz = c2d_tustin(num_Gc, den_Gc, 2.0)
    np.testing.assert_allclose(nz, [2.9319, 0.8352, 0.0595], atol=1e-3)
    np.testing.assert_allclose(dz, [1.0, 0.0, -1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Ex 6(a) — compensador a/(s+b)
# ---------------------------------------------------------------------------

def test_ex6a_apb():
    sd = -2.5 + 2.0j

    numG = [2.0, 2.0]
    denG = [1.0, 2.0, 2.0]
    numH = [1.0, 3.0]
    denH = [1.0, 5.0]

    r = projeto_compensador_apb(numG, denG, numH, denH, sd)
    assert abs(r["a"] - 3.7999) < 1e-3
    assert abs(r["b"] - 2.8061) < 1e-3

    nT, dT = malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
    assert _polo_mais_proximo(np.roots(dT), sd) < 1e-9


# ---------------------------------------------------------------------------
# Ex 6(b) — Euler T = 1 s sobre G·H·Gc
# ---------------------------------------------------------------------------

def test_ex6b_euler():
    sd = -2.5 + 2.0j

    numG = [2.0, 2.0]
    denG = [1.0, 2.0, 2.0]
    numH = [1.0, 3.0]
    denH = [1.0, 5.0]

    r = projeto_compensador_apb(numG, denG, numH, denH, sd)

    n_GH, d_GH = tf_mul(numG, denG, numH, denH)
    n_tot, d_tot = tf_mul(n_GH, d_GH, r["Gc_num"], r["Gc_den"])

    nz, dz = c2d_euler(n_tot, d_tot, 1.0)
    np.testing.assert_allclose(
        dz, [1.0, 5.806, 8.224, 5.806, 7.224], atol=1e-2
    )
