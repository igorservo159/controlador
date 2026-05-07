"""Núcleo numérico para projeto de controladores via LGR.

Convenções:
- Funções de transferência são pares ``(num, den)``, onde ``num`` e ``den``
  são arrays numpy de coeficientes em ordem decrescente de potência
  (convenção numpy/scipy).
- ``s`` é tratado como complexo. Polos/zeros idem.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import TransferFunction, step


# ---------------------------------------------------------------------------
# Manipulação de funções de transferência
# ---------------------------------------------------------------------------

def tf_eval(num, den, s):
    """Avalia ``num(s)/den(s)`` em ``s`` (real ou complexo)."""
    return np.polyval(num, s) / np.polyval(den, s)


def tf_mul(num1, den1, num2, den2):
    """Produto de duas TFs."""
    n1 = np.asarray(num1, dtype=float)
    d1 = np.asarray(den1, dtype=float)
    n2 = np.asarray(num2, dtype=float)
    d2 = np.asarray(den2, dtype=float)
    return np.convolve(n1, n2), np.convolve(d1, d2)


def feedback(numG, denG, numH, denH):
    """Realimentação negativa: ``T = G / (1 + G·H)``."""
    nG = np.asarray(numG, dtype=float)
    dG = np.asarray(denG, dtype=float)
    nH = np.asarray(numH, dtype=float)
    dH = np.asarray(denH, dtype=float)

    num_T = np.convolve(nG, dH)
    den_T = np.polyadd(np.convolve(dG, dH), np.convolve(nG, nH))
    return num_T, den_T


def malha_fechada(numG, denG, numH, denH, num_Gc, den_Gc):
    """Malha fechada com controlador em série e realimentação ``H``.

    ``T(s) = Gc·G / (1 + Gc·G·H)``.
    """
    n_GcG, d_GcG = tf_mul(num_Gc, den_Gc, numG, denG)
    return feedback(n_GcG, d_GcG, numH, denH)


# ---------------------------------------------------------------------------
# Especificações → polo dominante
# ---------------------------------------------------------------------------

def zeta_de_mp(Mp_pct):
    """ζ a partir do sobressinal percentual."""
    if Mp_pct <= 0 or Mp_pct >= 100:
        raise ValueError("Mp deve estar em (0, 100)")
    r = np.log(Mp_pct / 100.0)
    return -r / np.sqrt(np.pi**2 + r**2)


def wn_min(zeta, ts, criterio="2%"):
    """ωn mínimo para satisfazer ts (critério "2%" ou "5%")."""
    k = {"2%": 4.0, "5%": 3.0}[criterio]
    return k / (zeta * ts)


def polo_desejado(zeta, wn):
    """``sd = -ζωn + j·ωn·sqrt(1-ζ²)``."""
    if zeta >= 1.0:
        raise ValueError("ζ ≥ 1 produz polos reais; este projeto assume polos complexos")
    return -zeta * wn + 1j * wn * np.sqrt(1.0 - zeta**2)


# ---------------------------------------------------------------------------
# Helpers de ângulo
# ---------------------------------------------------------------------------

def _wrap_pm180(deg):
    """Reduz ``deg`` ao intervalo ``(-180, 180]``."""
    out = ((deg + 180.0) % 360.0) - 180.0
    if out == -180.0:
        out = 180.0
    return out


def _z_de_angulo(sd, alpha_deg):
    """Posição do zero ``-z`` (real) tal que ``∠(sd + z) = α``.

    Geometria: ``tan(α) = ω_d / (σ_d + z)``  →  ``z = ω_d/tan(α) - σ_d``.
    """
    if not (0.0 < alpha_deg < 180.0):
        raise ValueError(f"α = {alpha_deg:.4f}° fora de (0°, 180°)")
    return sd.imag / np.tan(np.radians(alpha_deg)) - sd.real


# ---------------------------------------------------------------------------
# Projeto por topologia
# ---------------------------------------------------------------------------

def projeto_P(numG, denG, numH, denH, sd):
    """``Gc(s) = Kc`` (proporcional puro).

    Para que ``sd`` esteja no LGR é necessário ``∠GH(sd) ≡ ±180°``.
    Caso o desvio seja maior que 1°, retorna o melhor ``Kc`` mas marca
    ``angular_ok = False``.
    """
    n_GH, d_GH = tf_mul(numG, denG, numH, denH)
    GH = tf_eval(n_GH, d_GH, sd)
    ang_GH = float(np.degrees(np.angle(GH)))
    M_GH = float(abs(GH))

    desvio = min(abs(_wrap_pm180(ang_GH - 180.0)),
                 abs(_wrap_pm180(ang_GH + 180.0)))
    Kc = 1.0 / M_GH

    return {
        "topologia": "P",
        "Gc_num": np.array([Kc]),
        "Gc_den": np.array([1.0]),
        "Kc": Kc,
        "Kp": Kc,
        "ang_GH": ang_GH,
        "M_GH": M_GH,
        "GH_sd": GH,
        "desvio_angular": desvio,
        "angular_ok": desvio < 1.0,
    }


def projeto_PD(numG, denG, numH, denH, sd):
    """``Gc(s) = Kc·(s + z) = Kp + Kd·s``."""
    n_GH, d_GH = tf_mul(numG, denG, numH, denH)
    GH = tf_eval(n_GH, d_GH, sd)
    ang_GH = float(np.degrees(np.angle(GH)))
    M_GH = float(abs(GH))

    alpha = _wrap_pm180(180.0 - ang_GH)
    if not (0.0 < alpha < 180.0):
        alpha = _wrap_pm180(-180.0 - ang_GH)

    z = _z_de_angulo(sd, alpha)
    Kc = 1.0 / (abs(sd + z) * M_GH)

    return {
        "topologia": "PD",
        "Gc_num": Kc * np.array([1.0, z]),
        "Gc_den": np.array([1.0]),
        "Kc": Kc,
        "z": z,
        "Kp": Kc * z,
        "Kd": Kc,
        "ang_GH": ang_GH,
        "M_GH": M_GH,
        "GH_sd": GH,
        "theta_c": alpha,
    }


def projeto_PI(numG, denG, numH, denH, sd):
    """``Gc(s) = Kc·(s + z)/s = Kp + Ki/s``."""
    n_GH, d_GH = tf_mul(numG, denG, numH, denH)
    GH = tf_eval(n_GH, d_GH, sd)
    ang_GH = float(np.degrees(np.angle(GH)))
    M_GH = float(abs(GH))
    ang_sd = float(np.degrees(np.angle(sd)))
    abs_sd = float(abs(sd))

    alpha = _wrap_pm180(180.0 - ang_GH + ang_sd)
    if not (0.0 < alpha < 180.0):
        alpha = _wrap_pm180(-180.0 - ang_GH + ang_sd)

    z = _z_de_angulo(sd, alpha)
    Kc = abs_sd / (abs(sd + z) * M_GH)

    return {
        "topologia": "PI",
        "Gc_num": Kc * np.array([1.0, z]),
        "Gc_den": np.array([1.0, 0.0]),
        "Kc": Kc,
        "z": z,
        "Kp": Kc,
        "Ki": Kc * z,
        "ang_GH": ang_GH,
        "M_GH": M_GH,
        "GH_sd": GH,
        "theta_c": alpha,
        "ang_sd": ang_sd,
    }


def projeto_PID_zigual(numG, denG, numH, denH, sd):
    """``Gc(s) = Kc·(s + z)² / s`` com ``z₁ = z₂ = z``.

    Equivale a ``Kp + Ki/s + Kd·s`` com ``Kd = Kc``, ``Kp = 2 Kc z``,
    ``Ki = Kc z²``.
    """
    n_GH, d_GH = tf_mul(numG, denG, numH, denH)
    GH = tf_eval(n_GH, d_GH, sd)
    ang_GH = float(np.degrees(np.angle(GH)))
    M_GH = float(abs(GH))
    ang_sd = float(np.degrees(np.angle(sd)))
    abs_sd = float(abs(sd))

    # 2α ≡ 180 - ang_GH + ang_sd (mod 360°), pois +180 e -180 diferem por 360.
    target_2alpha = (180.0 - ang_GH + ang_sd) % 360.0  # em [0, 360)
    alpha = target_2alpha / 2.0  # em [0, 180)
    if alpha <= 0.0 or alpha >= 180.0:
        raise ValueError(f"α resultante = {alpha:.4f}° não permite zero real estável")

    z = _z_de_angulo(sd, alpha)
    Kc = abs_sd / (abs(sd + z) ** 2 * M_GH)

    Gc_num = Kc * np.array([1.0, 2.0 * z, z * z])
    Gc_den = np.array([1.0, 0.0])

    return {
        "topologia": "PID_zigual",
        "Gc_num": Gc_num,
        "Gc_den": Gc_den,
        "Kc": Kc,
        "z1": z,
        "z2": z,
        "Kp": 2.0 * Kc * z,
        "Ki": Kc * z * z,
        "Kd": Kc,
        "ang_GH": ang_GH,
        "M_GH": M_GH,
        "GH_sd": GH,
        "theta_c": alpha,
        "ang_sd": ang_sd,
    }


def projeto_compensador_apb(numG, denG, numH, denH, sd):
    """Compensador ``Gc(s) = a / (s + b)`` (lead/lag genérico)."""
    n_GH, d_GH = tf_mul(numG, denG, numH, denH)
    GH = tf_eval(n_GH, d_GH, sd)
    ang_GH = float(np.degrees(np.angle(GH)))
    M_GH = float(abs(GH))

    # ∠Gc(sd) = -∠(sd+b); fechar 180° com ang_GH ⇒ ∠(sd+b) = ang_GH ∓ 180°
    beta = _wrap_pm180(ang_GH - 180.0)
    if not (0.0 < beta < 180.0):
        beta = _wrap_pm180(ang_GH + 180.0)

    b = _z_de_angulo(sd, beta)
    a = abs(sd + b) / M_GH

    return {
        "topologia": "apb",
        "Gc_num": np.array([a]),
        "Gc_den": np.array([1.0, b]),
        "a": a,
        "b": b,
        "ang_GH": ang_GH,
        "M_GH": M_GH,
        "GH_sd": GH,
        "theta_c": beta,
    }


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def metricas_step(num_T, den_T, t_max=None, n_pts=4000):
    """Mp, ts(2%), ts(5%), polos e resposta ao degrau de ``T(s)``.

    Mp é medido após a saída cruzar ``yss`` pela primeira vez (§2.6 do
    CLAUDE.md), evitando contar o salto inicial de TFs com grau de
    numerador igual ao do denominador.
    """
    num_T = np.asarray(num_T, dtype=float)
    den_T = np.asarray(den_T, dtype=float)
    polos = np.roots(den_T)

    if t_max is None:
        partes_neg = polos.real[polos.real < 0]
        if partes_neg.size == 0:
            t_max = 30.0
        else:
            tau_max = -1.0 / np.max(partes_neg)
            t_max = max(20.0 * tau_max, 5.0)

    t = np.linspace(0.0, t_max, n_pts)
    sys = TransferFunction(num_T, den_T)
    _, y = step(sys, T=t)

    yss = float(np.polyval(num_T, 0.0) / np.polyval(den_T, 0.0))

    Mp = 0.0
    if abs(yss) > 1e-12:
        y0 = float(y[0])
        if yss > y0:
            crosses = np.where(y >= yss)[0]
            if crosses.size > 0:
                ymax = float(np.max(y[crosses[0]:]))
                Mp = max((ymax - yss) / yss * 100.0, 0.0)
        elif yss < y0:
            crosses = np.where(y <= yss)[0]
            if crosses.size > 0:
                ymin = float(np.min(y[crosses[0]:]))
                Mp = max((yss - ymin) / abs(yss) * 100.0, 0.0)

    def _ts(tol):
        if abs(yss) < 1e-12:
            return 0.0
        fora = np.abs(y - yss) > tol * abs(yss)
        if not np.any(fora):
            return 0.0
        return float(t[np.where(fora)[0][-1]])

    return {
        "Mp": Mp,
        "ts2": _ts(0.02),
        "ts5": _ts(0.05),
        "yss": yss,
        "polos": polos,
        "t": t,
        "y": y,
    }


# ---------------------------------------------------------------------------
# Discretização
# ---------------------------------------------------------------------------

def _c2d_substituicao(num, den, T, metodo):
    """Substitui ``s`` por ``f(z)`` por substituição polinomial direta.

    Suporta TFs impróprias (ex.: ``Gc`` PID puro com 2 zeros e 1 polo).
    ``cont2discrete`` do scipy rejeita TFs impróprias — daí o método à mão.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    if metodo == "tustin":
        s_num = np.array([2.0 / T, -2.0 / T])  # (2/T)·(z - 1)
        s_den = np.array([1.0, 1.0])           # z + 1
    elif metodo == "euler":
        s_num = np.array([1.0 / T, -1.0 / T])  # (1/T)·(z - 1)
        s_den = np.array([1.0])                # 1
    else:
        raise ValueError(f"método desconhecido: {metodo!r}")

    def evaluate(coefs):
        n = len(coefs) - 1
        acc = np.array([0.0])
        for i, c in enumerate(coefs):
            p = n - i  # potência atual de s
            term = np.array([float(c)])
            for _ in range(p):
                term = np.convolve(term, s_num)
            for _ in range(n - p):
                term = np.convolve(term, s_den)
            acc = np.polyadd(acc, term)
        return acc, n

    nz, n_n = evaluate(num)
    dz, n_d = evaluate(den)

    n_max = max(n_n, n_d)
    for _ in range(n_max - n_n):
        nz = np.convolve(nz, s_den)
    for _ in range(n_max - n_d):
        dz = np.convolve(dz, s_den)

    while len(nz) > 1 and abs(nz[0]) < 1e-14:
        nz = nz[1:]
    while len(dz) > 1 and abs(dz[0]) < 1e-14:
        dz = dz[1:]

    lead = dz[0]
    return nz / lead, dz / lead


def c2d_tustin(num, den, T):
    """Bilinear: ``s ← (2/T)·(z - 1)/(z + 1)``."""
    return _c2d_substituicao(num, den, T, "tustin")


def c2d_euler(num, den, T):
    """Euler forward: ``s ← (z - 1)/T``."""
    return _c2d_substituicao(num, den, T, "euler")
