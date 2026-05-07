"""Aplicativo Streamlit para projeto de controladores via LGR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import lgr
import plotting
from presets import PRESETS

st.set_page_config(page_title="Projeto LGR", layout="wide")
st.title("Projeto de Controladores via Lugar das Raízes")
st.caption("DCA-3701 — UFRN. Suporta P, PD, PI, PID (z₁=z₂) e compensador a/(s+b).")


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def parse_poly(text):
    """Lê coeficientes separados por espaço ou vírgula em ordem decrescente."""
    text = text.replace(",", " ")
    return [float(t) for t in text.split() if t]


def _list_to_str(lst):
    return " ".join(f"{x:g}" for x in lst)


def fmt_num(x, dec=4):
    """Formata número para LaTeX, removendo zeros à direita supérfluos."""
    if isinstance(x, complex):
        if abs(x.imag) < 1e-10:
            return fmt_num(x.real, dec)
        sign = "+" if x.imag >= 0 else "-"
        return f"{x.real:.{dec}f} {sign} {abs(x.imag):.{dec}f}j"
    return f"{x:.{dec}g}"


def fmt_poly(coefs, var="s"):
    """Polinômio → string LaTeX."""
    coefs = list(coefs)
    n = len(coefs) - 1
    parts = []
    for i, c in enumerate(coefs):
        p = n - i
        if abs(c) < 1e-12:
            continue
        sign = "-" if c < 0 else ("+" if parts else "")
        c_abs = abs(c)
        if p == 0:
            term = f"{c_abs:.4g}"
        else:
            coef_str = "" if abs(c_abs - 1.0) < 1e-12 else f"{c_abs:.4g}\\,"
            pow_str = var if p == 1 else f"{var}^{{{p}}}"
            term = f"{coef_str}{pow_str}"
        if parts:
            parts.append(f" {sign} ")
        elif sign == "-":
            parts.append("-")
        parts.append(term)
    return "".join(parts) if parts else "0"


def fmt_tf(num, den, var="s"):
    """``num/den`` como fração LaTeX, ou apenas ``num`` se den=1."""
    if len(den) == 1 and abs(den[0] - 1.0) < 1e-12:
        return fmt_poly(num, var)
    return rf"\dfrac{{{fmt_poly(num, var)}}}{{{fmt_poly(den, var)}}}"


def fmt_complex(c, dec=4):
    """Complex number → string. Trata parte imaginária ~0 como real."""
    c = complex(c)
    if abs(c.imag) < 1e-8:
        return f"{c.real:.{dec}f}"
    sign = "+" if c.imag >= 0 else "-"
    return f"{c.real:.{dec}f} {sign} {abs(c.imag):.{dec}f}j"


def fmt_dist_label(root, dec=4):
    """Rótulo LaTeX para |s_d - root|, preferindo +/- compactos quando real."""
    root = complex(root)
    if abs(root.imag) < 1e-8:
        r = root.real
        if abs(r) < 1e-10:
            return r"|s_d|"
        if r < 0:
            return rf"|s_d + {(-r):.{dec}g}|"
        return rf"|s_d - {r:.{dec}g}|"
    return rf"|s_d - ({fmt_complex(root, dec)})|"


# ---------------------------------------------------------------------------
# Sidebar — preset selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Preset")
    nome = st.selectbox("Carregar exemplo", list(PRESETS.keys()))
    preset = PRESETS[nome]
    if preset:
        st.info(preset["descricao"])
        if st.button("Aplicar preset", use_container_width=True):
            st.session_state["numG_str"] = _list_to_str(preset["numG"])
            st.session_state["denG_str"] = _list_to_str(preset["denG"])
            st.session_state["numH_str"] = _list_to_str(preset["numH"])
            st.session_state["denH_str"] = _list_to_str(preset["denH"])
            st.session_state["topologia"] = preset["topologia"]
            st.session_state["modo_specs"] = preset["modo_specs"]
            if preset["modo_specs"] == "mp_ts":
                st.session_state["Mp"] = preset["Mp"]
                st.session_state["ts"] = preset["ts"]
                st.session_state["criterio_ts"] = preset["criterio_ts"]
            elif preset["modo_specs"] == "zeta_wn":
                st.session_state["zeta"] = preset["zeta"]
                st.session_state["wn"] = preset["wn"]
            elif preset["modo_specs"] == "polos":
                st.session_state["sd_re"] = preset["sd_re"]
                st.session_state["sd_im"] = preset["sd_im"]
            st.session_state["mult_Kc"] = 1.0
            st.session_state["discretizar"] = preset.get("discretizar", False)
            if preset.get("discretizar"):
                st.session_state["metodo_disc"] = preset["metodo_disc"]
                st.session_state["T_disc"] = preset["T"]
                st.session_state["discretizar_alvo"] = preset["discretizar_alvo"]
            st.rerun()
    st.markdown("---")
    st.markdown(
        "Coeficientes em **ordem decrescente** de potência, separados por "
        "espaço ou vírgula. Ex.: `1 4 4 0` para $s^3+4s^2+4s$."
    )


# ---------------------------------------------------------------------------
# Seção 1 — Planta e realimentação
# ---------------------------------------------------------------------------

st.header("1. Planta e realimentação")
col1, col2 = st.columns(2)
with col1:
    numG_str = st.text_input("Numerador G(s)", value="4 16", key="numG_str")
    denG_str = st.text_input("Denominador G(s)", value="1 4 4 0", key="denG_str")
with col2:
    numH_str = st.text_input("Numerador H(s)", value="1", key="numH_str")
    denH_str = st.text_input("Denominador H(s)", value="1", key="denH_str")

try:
    numG = parse_poly(numG_str)
    denG = parse_poly(denG_str)
    numH = parse_poly(numH_str)
    denH = parse_poly(denH_str)
    if not numG or not denG or not numH or not denH:
        raise ValueError("polinômios não podem ser vazios")
except Exception as e:
    st.error(f"Polinômios inválidos: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.latex(rf"G(s) = {fmt_tf(numG, denG)}")
with c2:
    st.latex(rf"H(s) = {fmt_tf(numH, denH)}")


# ---------------------------------------------------------------------------
# Seção 2 — Topologia
# ---------------------------------------------------------------------------

st.header("2. Tipo de controlador")
TOPOLOGIAS = ["P", "PD", "PI", "PID_zigual", "apb"]
TOPOLOGIA_LABEL = {
    "P": "P — proporcional",
    "PD": "PD — Kc·(s + z)",
    "PI": "PI — Kc·(s + z)/s",
    "PID_zigual": "PID com z₁ = z₂ — Kc·(s + z)²/s",
    "apb": "Compensador a/(s + b)",
}
topologia = st.selectbox(
    "Topologia",
    TOPOLOGIAS,
    format_func=lambda k: TOPOLOGIA_LABEL[k],
    key="topologia",
)


# ---------------------------------------------------------------------------
# Seção 3 — Especificações
# ---------------------------------------------------------------------------

st.header("3. Especificações de desempenho")
MODO_LABEL = {
    "mp_ts": "Mp + tempo de assentamento",
    "zeta_wn": "ζ + ωn diretos",
    "polos": "polo dominante sd diretamente",
}
modo = st.radio(
    "Modo",
    list(MODO_LABEL.keys()),
    format_func=lambda k: MODO_LABEL[k],
    key="modo_specs",
    horizontal=True,
)

if modo == "mp_ts":
    c1, c2, c3 = st.columns(3)
    Mp_pct = c1.number_input("Mp (%)", min_value=0.01, max_value=99.99,
                              value=10.0, step=0.5, key="Mp")
    ts = c2.number_input("ts (s)", min_value=0.01, value=4.0, step=0.1, key="ts")
    criterio_ts = c3.radio("Critério", ["2%", "5%"], horizontal=True,
                            key="criterio_ts")
    zeta = lgr.zeta_de_mp(Mp_pct)
    wn = lgr.wn_min(zeta, ts, criterio_ts)
    sd = lgr.polo_desejado(zeta, wn)

elif modo == "zeta_wn":
    c1, c2 = st.columns(2)
    zeta = c1.number_input("ζ", min_value=0.01, max_value=0.99,
                           value=0.7, step=0.01, key="zeta")
    wn = c2.number_input("ωn (rad/s)", min_value=0.001,
                         value=1.0, step=0.1, key="wn")
    Mp_pct = 100.0 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    sd = lgr.polo_desejado(zeta, wn)
    ts = None
    criterio_ts = "2%"

else:  # polos
    c1, c2 = st.columns(2)
    sd_re = c1.number_input("Re(sd)", value=-1.0, step=0.1, key="sd_re")
    sd_im = c2.number_input("Im(sd)", value=1.0, step=0.1, key="sd_im")
    sd = sd_re + 1j * sd_im
    if sd.imag <= 0:
        st.warning("Im(sd) deve ser positivo (o conjugado é considerado automaticamente).")
        st.stop()
    wn = abs(sd)
    zeta = -sd.real / wn
    if zeta <= 0:
        st.warning("Re(sd) deve ser negativo para projeto estável.")
    Mp_pct = 100.0 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) if 0 < zeta < 1 else 0.0
    ts = None
    criterio_ts = "2%"


# ---------------------------------------------------------------------------
# Seção 4 — Discretização (expander)
# ---------------------------------------------------------------------------

with st.expander("4. Discretização (opcional)", expanded=False):
    discretizar = st.checkbox("Discretizar controlador",
                               key="discretizar")
    metodo_disc = "tustin"
    T_disc = 1.0
    discretizar_alvo = "Gc"
    if discretizar:
        c1, c2, c3 = st.columns(3)
        metodo_disc = c1.radio(
            "Método", ["tustin", "euler"],
            format_func=lambda k: {"tustin": "Tustin (bilinear)",
                                    "euler": "Euler forward"}[k],
            key="metodo_disc",
        )
        T_disc = c2.number_input("Período T (s)", min_value=0.001,
                                  value=1.0, step=0.1, key="T_disc")
        discretizar_alvo = c3.radio(
            "Discretizar",
            ["Gc", "GHGc"],
            format_func=lambda k: {"Gc": "Gc(s)",
                                    "GHGc": "G(s)·H(s)·Gc(s)"}[k],
            key="discretizar_alvo",
        )


# ---------------------------------------------------------------------------
# Seção 5 — Multiplicador de Kc
# ---------------------------------------------------------------------------

st.header("5. Multiplicador de Kc")
mult_Kc = st.slider(
    "Multiplicador",
    min_value=0.1, max_value=3.0, value=1.0, step=0.05,
    help=("Ajuste fino. Reduza se o Mp da simulação estourar; aumente se "
          "ficou subamortecido demais."),
    key="mult_Kc",
)


st.markdown("---")


# ---------------------------------------------------------------------------
# Projeto
# ---------------------------------------------------------------------------

PROJETO = {
    "P": lgr.projeto_P,
    "PD": lgr.projeto_PD,
    "PI": lgr.projeto_PI,
    "PID_zigual": lgr.projeto_PID_zigual,
    "apb": lgr.projeto_compensador_apb,
}

try:
    r = PROJETO[topologia](numG, denG, numH, denH, sd)
except Exception as e:
    st.error(f"Erro no projeto: {e}")
    st.stop()

# Captura ganho de projeto antes do multiplicador (útil no Passo 5)
Kc_design = r["a"] if topologia == "apb" else r["Kc"]

# Aplicar multiplicador de Kc / ganho
if topologia == "apb":
    r["a"] *= mult_Kc
    r["Gc_num"] = np.array([r["a"]])
else:
    r["Kc"] *= mult_Kc
    if topologia == "P":
        r["Kp"] = r["Kc"]
        r["Gc_num"] = np.array([r["Kc"]])
    elif topologia == "PD":
        r["Kp"] = r["Kc"] * r["z"]
        r["Kd"] = r["Kc"]
        r["Gc_num"] = r["Kc"] * np.array([1.0, r["z"]])
    elif topologia == "PI":
        r["Kp"] = r["Kc"]
        r["Ki"] = r["Kc"] * r["z"]
        r["Gc_num"] = r["Kc"] * np.array([1.0, r["z"]])
    elif topologia == "PID_zigual":
        z = r["z1"]
        r["Kd"] = r["Kc"]
        r["Kp"] = 2.0 * r["Kc"] * z
        r["Ki"] = r["Kc"] * z * z
        r["Gc_num"] = r["Kc"] * np.array([1.0, 2 * z, z * z])


# ---------------------------------------------------------------------------
# Passos 1–6 — apresentação
# ---------------------------------------------------------------------------

with st.expander("Passo 1 — Especificações e polo dominante $s_d$", expanded=True):
    if modo == "mp_ts":
        st.latex(r"\zeta = \dfrac{-\ln(M_p/100)}{\sqrt{\pi^2 + \ln^2(M_p/100)}}")
        st.markdown(f"$M_p = {Mp_pct:.4g}\\%$  ⇒  $\\zeta = {zeta:.4f}$")
        k_eq = 4 if criterio_ts == "2%" else 3
        st.latex(rf"\omega_n^{{min}} = \dfrac{{{k_eq}}}{{\zeta\, t_s}}")
        st.markdown(
            f"$t_s({criterio_ts}) \\leq {ts:.4g}\\,s$  ⇒  "
            f"$\\omega_n^{{min}} = {wn:.4f}\\,$rad/s"
        )
    elif modo == "zeta_wn":
        st.markdown(f"$\\zeta = {zeta:.4f}$, $\\omega_n = {wn:.4f}\\,$rad/s")
        st.markdown(f"Sobressinal correspondente: $M_p \\approx {Mp_pct:.2f}\\%$")
    else:
        st.markdown(f"$s_d$ informado diretamente.  "
                    f"$\\zeta = {zeta:.4f}$, $\\omega_n = {wn:.4f}$.")

    st.latex(r"s_d = -\zeta\omega_n + j\,\omega_n\sqrt{1-\zeta^2}")
    st.markdown(f"$s_d = {sd.real:.4f} + {sd.imag:.4f}\\,j$")


with st.expander("Passo 2 — $G(s_d)\\cdot H(s_d)$ e decomposição angular", expanded=True):
    GH = r["GH_sd"]
    st.latex(r"G(s_d)\,H(s_d) = " +
             f"{GH.real:.4f}{'+' if GH.imag >= 0 else '-'}{abs(GH.imag):.4f}\\,j")
    st.markdown(
        f"Módulo: $|G\\cdot H|_{{s_d}} = {r['M_GH']:.4g}$  "
        f"&nbsp;&nbsp; Ângulo: $\\angle G\\cdot H |_{{s_d}} = {r['ang_GH']:.4f}^\\circ$"
    )

    # Decomposição angular da malha aberta Gc·G·H
    n_OL, d_OL = lgr.tf_mul(
        *lgr.tf_mul(numG, denG, numH, denH),
        r["Gc_num"], r["Gc_den"],
    )
    zeros_OL = np.roots(n_OL) if len(n_OL) > 1 else np.array([], dtype=complex)
    poles_OL = np.roots(d_OL) if len(d_OL) > 1 else np.array([], dtype=complex)
    K_lead_OL = float(n_OL[0]) / float(d_OL[0])

    st.markdown("**Decomposição angular** da malha aberta "
                "$G_c(s)\\,G(s)\\,H(s)$ — ângulo de cada vetor "
                "$s_d - \\text{singularidade}$:")
    rows = []
    soma_z, soma_p = 0.0, 0.0
    for z in zeros_OL:
        v = sd - z
        ang = float(np.degrees(np.angle(v)))
        soma_z += ang
        rows.append({
            "Tipo": "zero",
            "Valor": fmt_complex(z),
            "sd − valor": fmt_complex(v),
            "|sd − valor|": f"{abs(v):.4f}",
            "Ângulo (°)": f"{ang:+.4f}",
        })
    for p in poles_OL:
        v = sd - p
        ang = float(np.degrees(np.angle(v)))
        soma_p += ang
        rows.append({
            "Tipo": "polo",
            "Valor": fmt_complex(p),
            "sd − valor": fmt_complex(v),
            "|sd − valor|": f"{abs(v):.4f}",
            "Ângulo (°)": f"{ang:+.4f}",
        })
    df_decomp = pd.DataFrame(rows)
    st.dataframe(df_decomp, use_container_width=True, hide_index=True)

    ang_K_lead = 0.0 if K_lead_OL >= 0 else 180.0
    total_bruto = ang_K_lead + soma_z - soma_p
    total_norm = ((total_bruto + 180.0) % 360.0) - 180.0
    ang_OL_tf = float(np.degrees(np.angle(lgr.tf_eval(n_OL, d_OL, sd))))
    diff_circ = ((total_norm - ang_OL_tf + 180.0) % 360.0) - 180.0

    nota_lead = (
        f" + $\\angle K_g$ = {ang_K_lead:.0f}°"
        if K_lead_OL < 0 else ""
    )
    st.markdown(
        f"$\\sum$ ângulos zeros = **{soma_z:+.4f}°**, "
        f"$\\sum$ ângulos polos = **{soma_p:+.4f}°**"
        + (f", $K_g = {K_lead_OL:.4g} < 0$" if K_lead_OL < 0 else "")
    )
    st.markdown(
        f"**Total** = ($\\sum$z − $\\sum$p){nota_lead} = "
        f"{total_bruto:+.4f}° ≡ **{total_norm:+.4f}°** (mod 360°)"
    )
    st.markdown(
        f"Conferência via `tf_eval`: "
        f"$\\angle G_c G H |_{{s_d}} = {ang_OL_tf:+.4f}^\\circ$ "
        f"(diferença circular: {diff_circ:+.2e}°)"
    )


# Equação angular e cálculo do zero/polo
with st.expander("Passo 3 — Equação angular", expanded=True):
    if topologia == "P":
        st.latex(r"\angle [K_c \cdot G(s_d)\,H(s_d)] \equiv \pm 180^\circ")
        st.markdown(
            f"P puro só funciona se $\\angle GH(s_d)$ já está próximo de "
            f"$\\pm 180°$. Desvio atual: ${r['desvio_angular']:.2f}^\\circ$."
        )
        if not r["angular_ok"]:
            st.warning("$s_d$ não está sobre o LGR de $G\\cdot H$. P só "
                       "pode mover o polo dominante ao longo do LGR — "
                       "considere PD/PI/PID.")
    elif topologia == "PD":
        st.latex(r"\angle (s_d + z) = \pm 180^\circ - \angle G(s_d)\,H(s_d)")
        st.markdown(f"$\\theta_c = {r['theta_c']:.4f}^\\circ$")
    elif topologia == "PI":
        st.latex(r"\angle (s_d + z) = \pm 180^\circ - \angle GH(s_d) + \angle s_d")
        st.markdown(
            f"$\\angle s_d = {r['ang_sd']:.4f}^\\circ$, "
            f"$\\theta_c = {r['theta_c']:.4f}^\\circ$"
        )
    elif topologia == "PID_zigual":
        st.latex(r"2\,\angle (s_d + z) = \pm 180^\circ - \angle GH(s_d) + \angle s_d")
        st.markdown(
            f"$\\angle s_d = {r['ang_sd']:.4f}^\\circ$, "
            f"$\\theta_c = {r['theta_c']:.4f}^\\circ$ (cada zero)"
        )
    elif topologia == "apb":
        st.latex(r"\angle (s_d + b) = \angle GH(s_d) \mp 180^\circ")
        st.markdown(f"$\\angle (s_d+b) = {r['theta_c']:.4f}^\\circ$")


with st.expander("Passo 4 — Posição do zero/polo do controlador", expanded=True):
    if topologia == "PD":
        st.latex(r"z = \dfrac{\omega_d}{\tan\theta_c} - \sigma_d")
        st.markdown(f"$z = {r['z']:.4f}$")
        if r["z"] < 0:
            st.warning("$z < 0$: zero no semiplano direito (não-mínima fase).")
    elif topologia == "PI":
        st.markdown(f"$z = {r['z']:.4f}$")
        if r["z"] < 0:
            st.warning("$z < 0$: zero no semiplano direito.")
    elif topologia == "PID_zigual":
        st.markdown(f"$z_1 = z_2 = {r['z1']:.4f}$")
        if r["z1"] < 0:
            st.warning("$z < 0$: zero no semiplano direito.")
    elif topologia == "apb":
        st.markdown(f"$b = {r['b']:.4f}$")
        if r["b"] < 0:
            st.warning("$b < 0$: polo no semiplano direito (compensador instável).")
    else:
        st.markdown("Sem zero/polo extra a colocar (apenas ganho).")


with st.expander("Passo 5 — Ganho", expanded=True):
    nome_K = "a" if topologia == "apb" else "K_c"

    # ---- Forma fatorada (LGR) — pelo critério do módulo ----
    if topologia == "P":
        ctrl_num, ctrl_den = np.array([1.0]), np.array([1.0])
    elif topologia == "PD":
        ctrl_num, ctrl_den = np.array([1.0, r["z"]]), np.array([1.0])
    elif topologia == "PI":
        ctrl_num, ctrl_den = np.array([1.0, r["z"]]), np.array([1.0, 0.0])
    elif topologia == "PID_zigual":
        zz = r["z1"]
        ctrl_num = np.array([1.0, 2.0 * zz, zz * zz])
        ctrl_den = np.array([1.0, 0.0])
    elif topologia == "apb":
        ctrl_num, ctrl_den = np.array([1.0]), np.array([1.0, r["b"]])

    n_L, d_L = lgr.tf_mul(
        *lgr.tf_mul(numG, denG, numH, denH),
        ctrl_num, ctrl_den,
    )
    zeros_L = np.roots(n_L) if len(n_L) > 1 else np.array([], dtype=complex)
    poles_L = np.roots(d_L) if len(d_L) > 1 else np.array([], dtype=complex)
    K_lead_L = float(n_L[0]) / float(d_L[0])

    st.markdown("**Forma fatorada (critério do módulo do LGR):**")
    if abs(K_lead_L - 1.0) < 1e-10:
        st.latex(rf"{nome_K} = "
                 r"\dfrac{\prod_j |s_d + p_j|}{\prod_k |s_d + z_k|}")
    else:
        st.latex(rf"{nome_K} = "
                 r"\dfrac{\prod_j |s_d + p_j|}"
                 r"{K_g \cdot \prod_k |s_d + z_k|}"
                 rf",\quad K_g = {K_lead_L:.4g}")

    # Distâncias |sd - sing| para cada polo/zero
    zeros_dist = [abs(sd - z) for z in zeros_L]
    polos_dist = [abs(sd - p) for p in poles_L]

    if zeros_L.size:
        linhas_z = "  \n".join(
            f"&nbsp;&nbsp;${fmt_dist_label(z)} = {abs(sd - z):.4f}$"
            for z in zeros_L
        )
        st.markdown(f"Zeros de $L = G_c\\,G\\,H / {nome_K}$:  \n" + linhas_z)
    else:
        st.markdown(f"$L = G_c\\,G\\,H / {nome_K}$ não tem zeros finitos.")

    if poles_L.size:
        linhas_p = "  \n".join(
            f"&nbsp;&nbsp;${fmt_dist_label(p)} = {abs(sd - p):.4f}$"
            for p in poles_L
        )
        st.markdown(f"Polos de $L$:  \n" + linhas_p)
    else:
        st.markdown("$L$ não tem polos finitos.")

    prod_pol = float(np.prod(polos_dist)) if polos_dist else 1.0
    prod_zer = float(np.prod(zeros_dist)) if zeros_dist else 1.0

    if polos_dist:
        s_pol = " \\cdot ".join(f"{d:.4f}" for d in polos_dist)
        st.latex(rf"\textstyle\prod_j |s_d + p_j| = {s_pol} = {prod_pol:.4f}")
    if zeros_dist:
        s_zer = " \\cdot ".join(f"{d:.4f}" for d in zeros_dist)
        st.latex(rf"\textstyle\prod_k |s_d + z_k| = {s_zer} = {prod_zer:.4f}")

    Kc_factor = prod_pol / (K_lead_L * prod_zer)
    if abs(K_lead_L - 1.0) < 1e-10:
        st.latex(rf"{nome_K} = \dfrac{{{prod_pol:.4f}}}{{{prod_zer:.4f}}} "
                 rf"= {Kc_factor:.6g}")
    else:
        st.latex(rf"{nome_K} = \dfrac{{{prod_pol:.4f}}}"
                 rf"{{{K_lead_L:.4g} \cdot {prod_zer:.4f}}} "
                 rf"= {Kc_factor:.6g}")

    if abs(Kc_factor - Kc_design) > 1e-6 * max(abs(Kc_design), 1.0):
        st.warning(
            f"Discrepância vs. via tf_eval: fatorado = {Kc_factor:.6g}, "
            f"projeto = {Kc_design:.6g}."
        )

    st.markdown("---")

    # ---- Forma compacta original (via tf_eval) ----
    st.markdown("**Forma compacta** (via $|G\\cdot H|_{s_d}$ direto):")
    if topologia == "apb":
        st.latex(r"a = \dfrac{|s_d + b|}{|G(s_d)\,H(s_d)|}")
        st.markdown(f"$a = {r['a']:.6g}$"
                    + (f" (× mult. {mult_Kc:.2f})" if mult_Kc != 1.0 else ""))
    else:
        if topologia == "P":
            st.latex(r"K_c = \dfrac{1}{|G(s_d)\,H(s_d)|}")
        elif topologia == "PD":
            st.latex(r"K_c = \dfrac{1}{|s_d+z|\cdot |GH(s_d)|}")
        elif topologia == "PI":
            st.latex(r"K_c = \dfrac{|s_d|}{|s_d+z|\cdot |GH(s_d)|}")
        elif topologia == "PID_zigual":
            st.latex(r"K_c = \dfrac{|s_d|}{|s_d+z|^2\cdot |GH(s_d)|}")
        st.markdown(f"$K_c = {r['Kc']:.6g}$"
                    + (f" (× mult. {mult_Kc:.2f})" if mult_Kc != 1.0 else ""))


with st.expander("Passo 6 — Equação final do controlador", expanded=True):
    st.latex(rf"G_c(s) = {fmt_tf(r['Gc_num'], r['Gc_den'])}")
    if topologia == "P":
        st.markdown(f"$K_p = {r['Kp']:.4g}$")
    elif topologia == "PD":
        st.markdown(
            f"$K_p = K_c \\cdot z = {r['Kp']:.4g}$, "
            f"$K_d = K_c = {r['Kd']:.4g}$"
        )
    elif topologia == "PI":
        st.markdown(
            f"$K_p = K_c = {r['Kp']:.4g}$, "
            f"$K_i = K_c \\cdot z = {r['Ki']:.4g}$"
        )
    elif topologia == "PID_zigual":
        st.markdown(
            f"$K_p = 2K_c z = {r['Kp']:.4g}$, "
            f"$K_i = K_c z^2 = {r['Ki']:.4g}$, "
            f"$K_d = K_c = {r['Kd']:.4g}$"
        )
    elif topologia == "apb":
        st.markdown(f"$a = {r['a']:.4g}$, $b = {r['b']:.4g}$")


# ---------------------------------------------------------------------------
# Passo 7 — Validação gráfica
# ---------------------------------------------------------------------------

n_T, d_T = lgr.malha_fechada(numG, denG, numH, denH, r["Gc_num"], r["Gc_den"])
m = lgr.metricas_step(n_T, d_T)

st.markdown("### Passo 7 — Validação")
g1, g2, g3 = st.columns(3)
with g1:
    fig1 = plotting.fig_resposta_degrau(
        m["t"], m["y"], m["yss"], m["Mp"],
        m["ts2"] if criterio_ts == "2%" else m["ts5"],
        criterio_ts, titulo="Resposta ao degrau",
    )
    st.pyplot(fig1, use_container_width=True)
with g2:
    fig2 = plotting.fig_polos_zeros(n_T, d_T, sd=sd,
                                     titulo="Polos/zeros — MF")
    st.pyplot(fig2, use_container_width=True)
with g3:
    # LGR de L(s) = (Gc sem ganho ajustável) · G · H
    if topologia == "apb":
        # L(s) = G·H/(s+b); ganho ajustável é "a"
        n_L, d_L = lgr.tf_mul(np.array([1.0]), np.array([1.0, r["b"]]),
                               *lgr.tf_mul(numG, denG, numH, denH))
        K_design = r["a"]
    elif topologia == "P":
        n_L, d_L = lgr.tf_mul(numG, denG, numH, denH)
        K_design = r["Kc"]
    else:
        # PD, PI, PID — usa zeros do controlador, retira Kc
        if topologia == "PD":
            ctrl_num = np.array([1.0, r["z"]])
            ctrl_den = np.array([1.0])
        elif topologia == "PI":
            ctrl_num = np.array([1.0, r["z"]])
            ctrl_den = np.array([1.0, 0.0])
        else:  # PID_zigual
            z = r["z1"]
            ctrl_num = np.array([1.0, 2 * z, z * z])
            ctrl_den = np.array([1.0, 0.0])
        n_L, d_L = lgr.tf_mul(*lgr.tf_mul(ctrl_num, ctrl_den, numG, denG),
                               numH, denH)
        K_design = r["Kc"]
    fig3 = plotting.fig_lgr(n_L, d_L, K_design=K_design, sd=sd,
                             titulo="LGR")
    st.pyplot(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Passo 8 — Tabela "obtido vs especificado"
# ---------------------------------------------------------------------------

with st.expander("Passo 8 — Comparação obtido × especificado", expanded=True):
    polos = m["polos"]
    polo_perto = polos[np.argmin(np.abs(polos - sd))]
    zeta_obt = -polo_perto.real / abs(polo_perto) if abs(polo_perto) > 0 else 0
    wn_obt = abs(polo_perto)

    rows = [
        ["ζ", f"{zeta_obt:.4f}", f"{zeta:.4f}"],
        ["ωn (rad/s)", f"{wn_obt:.4f}", f"{wn:.4f}"],
        ["Mp (%)", f"{m['Mp']:.3f}",
         f"{Mp_pct:.3f}" if modo != "polos" else "(implícito)"],
    ]
    if criterio_ts == "2%":
        rows.append([f"ts(2%) (s)", f"{m['ts2']:.4f}",
                     f"{ts:.4f}" if ts is not None else "—"])
    else:
        rows.append([f"ts(5%) (s)", f"{m['ts5']:.4f}",
                     f"{ts:.4f}" if ts is not None else "—"])
    rows.append(["yss", f"{m['yss']:.4f}", "—"])
    rows.append(["Polo MF mais perto de sd",
                 f"{polo_perto.real:.4f} {'+' if polo_perto.imag >= 0 else '-'} {abs(polo_perto.imag):.4f}j",
                 f"{sd.real:.4f} + {sd.imag:.4f}j"])
    df = pd.DataFrame(rows, columns=["Grandeza", "Obtido", "Especificado"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(f"**Polos de malha fechada:** "
                + ", ".join(
                    f"{p.real:.4f}{'+' if p.imag >= 0 else '-'}{abs(p.imag):.4f}j"
                    if abs(p.imag) > 1e-8 else f"{p.real:.4f}"
                    for p in polos))


# ---------------------------------------------------------------------------
# Passo 9 — Discretização
# ---------------------------------------------------------------------------

if discretizar:
    st.markdown("### Passo 9 — Discretização")
    if discretizar_alvo == "Gc":
        n_in, d_in = r["Gc_num"], r["Gc_den"]
        rotulo_in = r"G_c(s)"
        rotulo_out = r"G_c(z)"
    else:  # GHGc
        n_GH, d_GH = lgr.tf_mul(numG, denG, numH, denH)
        n_in, d_in = lgr.tf_mul(n_GH, d_GH, r["Gc_num"], r["Gc_den"])
        rotulo_in = r"G(s)\,H(s)\,G_c(s)"
        rotulo_out = r"G(z)\,H(z)\,G_c(z)"

    if metodo_disc == "tustin":
        nz, dz = lgr.c2d_tustin(n_in, d_in, T_disc)
        st.markdown(f"Tustin com $T = {T_disc:.4g}$s : "
                    r"$s \leftarrow \dfrac{2}{T}\dfrac{z-1}{z+1}$")
    else:
        nz, dz = lgr.c2d_euler(n_in, d_in, T_disc)
        st.markdown(f"Euler forward com $T = {T_disc:.4g}$s : "
                    r"$s \leftarrow \dfrac{z-1}{T}$")

    st.latex(rf"{rotulo_out} = {fmt_tf(nz, dz, var='z')}")

    # Equação de diferenças (apenas se for próprio: deg num <= deg den)
    if len(nz) <= len(dz):
        # Normaliza para a₀ = 1 (já foi feito) e expressa
        # u[k] = -a₁ u[k-1] - a₂ u[k-2] - ... + b₀ e[k] + b₁ e[k-1] + ...
        # onde u é a saída do TF (Gc·) e e é a entrada (erro).
        # dz: [a₀=1, a₁, a₂, ...] (potências decrescentes em z)
        # nz: [b_m, ..., b₀] alinhado de modo que o índice 0 é o maior atraso
        # Ajusta padding para mesmo tamanho
        n_pad = np.concatenate([np.zeros(len(dz) - len(nz)), nz])
        partes_lhs = ["u[k]"]
        partes_rhs = []
        # termos do denominador (k-i) para i >= 1
        for i in range(1, len(dz)):
            ai = dz[i]
            if abs(ai) < 1e-12:
                continue
            sign = "-" if ai > 0 else "+"
            partes_rhs.append(f"{sign} {abs(ai):.4g}\\,u[k-{i}]")
        # termos do numerador
        for i, b in enumerate(n_pad):
            if abs(b) < 1e-12:
                continue
            sign = "+" if b > 0 else "-"
            atraso = i
            ek = "e[k]" if atraso == 0 else f"e[k-{atraso}]"
            if not partes_rhs and sign == "+":
                partes_rhs.append(f"{abs(b):.4g}\\,{ek}")
            else:
                partes_rhs.append(f"{sign} {abs(b):.4g}\\,{ek}")
        st.markdown("**Equação de diferenças:**")
        st.latex("u[k] = " + " ".join(partes_rhs))
    else:
        st.info("TF discretizada é imprópria (mais zeros que polos). "
                 "Pulando equação de diferenças.")
