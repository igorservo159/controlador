"""Funções de gráfico para o app Streamlit.

Cada função retorna uma ``matplotlib.figure.Figure`` para que ``app.py``
possa passá-la a ``st.pyplot``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Resposta ao degrau
# ---------------------------------------------------------------------------

def fig_resposta_degrau(t, y, yss, Mp, ts, criterio="2%", titulo=None):
    """Resposta ao degrau de malha fechada com bandas de tolerância e
    anotações de Mp e ts.
    """
    tol = 0.02 if criterio == "2%" else 0.05
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=100)
    ax.plot(t, y, lw=1.6, color="#1f77b4", label="y(t)")
    ax.axhline(yss, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.axhline(yss * (1 + tol), color="gray", lw=0.5, ls=":", alpha=0.7)
    ax.axhline(yss * (1 - tol), color="gray", lw=0.5, ls=":", alpha=0.7)

    # Anotações
    info = f"$M_p$ = {Mp:.2f}%\n$t_s$({criterio}) = {ts:.2f}s\n$y_{{ss}}$ = {yss:.3f}"
    ax.text(
        0.97, 0.05, info,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"),
    )

    ax.set_xlabel("t (s)")
    ax.set_ylabel("y(t)")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Mapa de polos e zeros de malha fechada
# ---------------------------------------------------------------------------

def fig_polos_zeros(num_T, den_T, sd=None, titulo=None):
    """Mapa polo-zero de ``T(s) = num_T/den_T``. Se ``sd`` for dado,
    marca-o (e seu conjugado) em destaque.
    """
    polos = np.roots(den_T) if len(den_T) > 1 else np.array([])
    zeros = np.roots(num_T) if len(num_T) > 1 else np.array([])

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=100)
    if zeros.size:
        ax.scatter(zeros.real, zeros.imag, marker="o", s=60,
                   facecolors="none", edgecolors="#2ca02c", lw=1.4, label="zeros MF")
    if polos.size:
        ax.scatter(polos.real, polos.imag, marker="x", s=60,
                   color="#d62728", lw=1.6, label="polos MF")
    if sd is not None:
        ax.scatter([sd.real, sd.real], [sd.imag, -sd.imag],
                   marker="*", s=120, color="#1f77b4",
                   edgecolors="black", lw=0.6, label="$s_d$ projetado")

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)

    pts = list(polos) + list(zeros)
    if sd is not None:
        pts += [sd, np.conj(sd)]
    if pts:
        re = np.array([p.real for p in pts])
        im = np.array([p.imag for p in pts])
        margin_re = max(0.5, 0.2 * (re.max() - re.min() + 1e-9))
        margin_im = max(0.5, 0.2 * (im.max() - im.min() + 1e-9))
        ax.set_xlim(re.min() - margin_re, re.max() + margin_re)
        ax.set_ylim(im.min() - margin_im, im.max() + margin_im)

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. LGR esboçado
# ---------------------------------------------------------------------------

def fig_lgr(num_L, den_L, K_design=None, sd=None, titulo=None,
           K_min=1e-2, K_max=1e3, n_K=300):
    """Esboço do LGR de ``1 + K·L(s) = 0`` para ``K`` em ``[K_min, K_max]``
    (escala log). Marca o ganho de projeto ``K_design`` se dado.

    Inclui também a malha aberta (zeros e polos de L(s)) e o ponto ``sd``.
    """
    num_L = np.asarray(num_L, dtype=float)
    den_L = np.asarray(den_L, dtype=float)

    # Polos e zeros da malha aberta
    polos_L = np.roots(den_L) if len(den_L) > 1 else np.array([])
    zeros_L = np.roots(num_L) if len(num_L) > 1 else np.array([])

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=100)

    # Varre K em escala log
    Ks = np.logspace(np.log10(K_min), np.log10(K_max), n_K)
    todas_raizes = []
    for K in Ks:
        # 1 + K·num/den = 0  ⇒  den + K·num = 0
        # Alinha graus para soma:
        a = np.zeros(max(len(den_L), len(num_L)))
        a[-len(den_L):] += den_L
        a[-len(num_L):] += K * num_L
        r = np.roots(a)
        todas_raizes.append(r)
    todas_raizes = np.concatenate(todas_raizes)

    ax.scatter(todas_raizes.real, todas_raizes.imag,
               s=2, color="#1f77b4", alpha=0.35, label="LGR")

    if zeros_L.size:
        ax.scatter(zeros_L.real, zeros_L.imag, marker="o", s=60,
                   facecolors="none", edgecolors="#2ca02c", lw=1.4, label="zeros MA")
    if polos_L.size:
        ax.scatter(polos_L.real, polos_L.imag, marker="x", s=60,
                   color="#d62728", lw=1.6, label="polos MA")

    if K_design is not None:
        a = np.zeros(max(len(den_L), len(num_L)))
        a[-len(den_L):] += den_L
        a[-len(num_L):] += K_design * num_L
        r_design = np.roots(a)
        ax.scatter(r_design.real, r_design.imag, marker="s", s=70,
                   color="#ff7f0e", edgecolors="black", lw=0.6,
                   label=f"K = {K_design:.4g}")
    if sd is not None:
        ax.scatter([sd.real, sd.real], [sd.imag, -sd.imag],
                   marker="*", s=130, color="#9467bd",
                   edgecolors="black", lw=0.6, label="$s_d$")

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, alpha=0.3)

    # Limites razoáveis: foca no entorno dos polos/zeros + sd
    pts_ref = list(polos_L) + list(zeros_L)
    if sd is not None:
        pts_ref += [sd, np.conj(sd)]
    if pts_ref:
        re = np.array([p.real for p in pts_ref])
        im = np.array([p.imag for p in pts_ref])
        re_range = max(re.max() - re.min(), 1.0)
        im_range = max(im.max() - im.min(), 1.0)
        ax.set_xlim(re.min() - 0.4 * re_range, re.max() + 0.4 * re_range)
        ax.set_ylim(im.min() - 0.4 * im_range, im.max() + 0.4 * im_range)

    fig.tight_layout()
    return fig
