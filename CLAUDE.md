# CLAUDE.md — App didático para projeto de controladores via LGR

## Objetivo

Construir um aplicativo **Streamlit** que projeta controladores
(P, PD, PI, PID com `z₁=z₂` e compensador `a/(s+b)`) pelo **método do Lugar
Geométrico das Raízes**, exatamente como ensinado na apostila *Sistemas de
Controle* (Fábio Araújo / UFRN, capítulo 4).

O app deve servir como **ferramenta genérica** — aceita qualquer planta
`G(s)`, qualquer realimentação `H(s)` e qualquer especificação (`Mp + ts`,
`ζ + ωn`, ou polo `sd` direto) — e mostrar o cálculo passo a passo. Os 6
exercícios da lista DCA-3701 2026.1 são **testes de regressão** com
respostas conhecidas (§4 abaixo).

---

## 1. Stack

```
streamlit
numpy
scipy
matplotlib
```

**Não use `python-control`.** O método é "à mão" via critérios de ângulo e
módulo — `python-control` esconderia justamente o que o app deve mostrar.
Toda manipulação de TFs é feita com polinômios numpy, e simulação no tempo
com `scipy.signal.TransferFunction` + `scipy.signal.step`.

---

## 2. Fundamento matemático

### 2.1 Especificações → polo dominante `sd`

Dados `Mp` (em %) e `ts` (em s):

```
ζ  = -ln(Mp/100) / sqrt(π² + ln²(Mp/100))
ωn = k / (ζ·ts)            com k=4 para ts(2%) e k=3 para ts(5%)
sd = -ζ·ωn + j·ωn·sqrt(1-ζ²)
```

Quando `ts < T` é apenas limite superior, qualquer `ωn ≥ ωn,min` serve.
O app usa `ωn = ωn,min` por padrão. Se o usuário quiser projetar com mais
margem, basta usar o modo `ζ + ωn` e escolher um `ωn` maior.

### 2.2 Critérios em `s = sd`

- **Ângulo**: `∠[Gc(sd)·G(sd)·H(sd)] ≡ ±180° (mod 360°)`
- **Módulo**: `|Gc(sd)·G(sd)·H(sd)| = 1`

O ângulo dá a posição do(s) zero(s)/polo(s) ajustável(is) do controlador;
o módulo dá o ganho `Kc`.

### 2.3 Geometria do zero real

Para um zero real em `s = -z` com `z > 0` e `sd = σ_d + j·ω_d` (com `ω_d > 0`):

```
∠(sd + z) = atan2(ω_d, σ_d + z)
```

Resolvendo para `z` dado um ângulo `α`:

```
z = ω_d / tan(α°) - σ_d        (com 0° < α < 180°)
```

### 2.4 Equações por topologia

Seja `θ_GH = ∠[G(sd)·H(sd)]` e `M_GH = |G(sd)·H(sd)|`.

| Topologia | Forma | Eq. ângulo (resolver) | Eq. módulo |
|---|---|---|---|
| **PD** | `Kc·(s+z)` | `∠(sd+z) = ±180° − θ_GH` | `Kc = 1/[\|sd+z\| · M_GH]` |
| **PI** | `Kc·(s+z)/s` | `∠(sd+z) = ±180° − θ_GH + ∠sd` | `Kc = \|sd\|/[\|sd+z\| · M_GH]` |
| **PID, z₁=z₂=z** | `Kc·(s+z)²/s` | `2·∠(sd+z) = ±180° − θ_GH + ∠sd` | `Kc = \|sd\|/[\|sd+z\|² · M_GH]` |
| **a/(s+b)** | `a/(s+b)` | `∠(sd+b) = θ_GH ∓ 180°` | `a = \|sd+b\|/M_GH` |

> **Escolha do ramo (±180°):** o ângulo resultante para o zero deve estar
> em `(0°, 180°)` quando `Im(sd) > 0`. Calcule os dois ramos e escolha o
> que satisfaz. Em código:
>
> ```python
> theta = ((alvo + 180) % 360) - 180
> if not (0 < theta < 180):
>     theta = ((alvo - 180 + 360) % 360) - 180
> ```

### 2.5 Identificação dos ganhos do PID padrão

A partir de `Gc(s) = Kp + Ki/s + Kd·s`:

- **PD** (`Kc(s+z) = Kp + Kd·s`): `Kp = Kc·z`, `Kd = Kc`.
- **PI** (`Kc(s+z)/s = Kp + Ki/s`): `Kp = Kc`, `Ki = Kc·z`.
- **PID, z₁=z₂=z** (expandindo `(s+z)² = s² + 2zs + z²`):
  - `Kd = Kc`, `Kp = 2·Kc·z`, `Ki = Kc·z²`

### 2.6 Cálculo de Mp em simulação — IMPORTANTE

Quando `T(s)` tem grau de numerador igual ao do denominador (ex.: PI com
`H(s)` de fase mínima), a saída do degrau pode arrancar com `y(0⁺) ≠ 0`.
O `max(y)` ocorre logo no início, mas isso **não é sobressinal**, é o
ganho de alta frequência.

**Defina Mp como o pico medido APÓS a saída cruzar `yss` pela primeira vez**:

```python
y0 = y[0]
if yss > y0:
    crosses = np.where(y >= yss)[0]
    if len(crosses) == 0:
        Mp = 0.0
    else:
        ymax = np.max(y[crosses[0]:])
        Mp = max((ymax - yss) / yss * 100.0, 0.0)
# análogo para yss < y0
```

### 2.7 Discretização

- **Tustin (bilinear)**: `s ← (2/T)·(z−1)/(z+1)`
- **Euler forward**: `s ← (z−1)/T`

`scipy.signal.cont2discrete` **rejeita TFs impróprias**. Como `Gc(s)` PID
puro (`Kc·(s+z)²/s`) é impróprio (grau 2 no num, 1 no den), implemente
substituição polinomial direta:

```python
def c2d_substituicao(num, den, T, metodo):
    """s ← f(z) por substituição polinomial. Suporta TFs impróprias."""
    if metodo == "tustin":
        s_num, s_den = np.array([2/T, -2/T]), np.array([1.0, 1.0])
    elif metodo == "euler":
        s_num, s_den = np.array([1/T, -1/T]), np.array([1.0])

    def evaluate(coefs):
        n = len(coefs) - 1
        acc = np.zeros(1)
        for i, c in enumerate(coefs):
            p = n - i
            term = np.array([c])
            for _ in range(p):     term = np.convolve(term, s_num)
            for _ in range(n - p): term = np.convolve(term, s_den)
            acc = np.polyadd(acc, term)
        return acc, n

    nz, n_n = evaluate(num)
    dz, n_d = evaluate(den)
    n_max = max(n_n, n_d)
    for _ in range(n_max - n_n): nz = np.convolve(nz, s_den)
    for _ in range(n_max - n_d): dz = np.convolve(dz, s_den)
    while len(nz) > 1 and abs(nz[0]) < 1e-14: nz = nz[1:]
    while len(dz) > 1 and abs(dz[0]) < 1e-14: dz = dz[1:]
    return nz / dz[0], dz / dz[0]
```

---

## 3. Núcleo numérico — `lgr.py`

Crie `lgr.py` com (no mínimo) estas funções. TFs sempre como `(num, den)` —
arrays de coeficientes em **ordem decrescente de potência** (convenção
numpy/scipy).

```python
# Manipulação de TFs
tf_eval(num, den, s)
tf_mul(num1, den1, num2, den2)
feedback(numG, denG, numH, denH)        # T = G/(1+GH), trata H ≠ 1
malha_fechada(numG, denG, numH, denH, num_Gc, den_Gc)

# Especificações
zeta_de_mp(Mp_pct)
wn_min(zeta, ts, criterio)              # criterio in {"2%", "5%"}
polo_desejado(zeta, wn)

# Projeto (cada um retorna dict: Gc_num, Gc_den, Kc, ganhos, ...)
projeto_PD(numG, denG, numH, denH, sd)
projeto_PI(numG, denG, numH, denH, sd)
projeto_PID_zigual(numG, denG, numH, denH, sd)
projeto_compensador_apb(numG, denG, numH, denH, sd)

# Validação
metricas_step(num_T, den_T, t_max=None) # → Mp, ts2, ts5, polos, t, y

# Discretização
c2d_tustin(num, den, T)
c2d_euler(num, den, T)
```

### 3.1 Esqueleto das funções de projeto

```python
def projeto_PD(numG, denG, numH, denH, sd):
    GH = tf_eval(*tf_mul(numG, denG, numH, denH), sd)
    ang_GH = np.degrees(np.angle(GH))

    theta_c = ((180.0 - ang_GH + 180) % 360) - 180
    if not (0 < theta_c < 180):
        theta_c = ((-180.0 - ang_GH + 180) % 360) - 180

    z = sd.imag / np.tan(np.radians(theta_c)) - sd.real
    Kc = 1.0 / (abs(sd + z) * abs(GH))

    return {
        "Gc_num": Kc * np.array([1.0, z]),
        "Gc_den": np.array([1.0]),
        "Kc": Kc, "z": z, "Kp": Kc*z, "Kd": Kc,
        "ang_GH": ang_GH, "theta_c": theta_c, "GH_sd": GH,
    }
```

Análogo para PI (denominador `[1, 0]`, ganho `|sd|/[|sd+z|·M_GH]`),
PID `z₁=z₂` (zeros duplos, ganho `|sd|/[|sd+z|²·M_GH]`) e a/(s+b)
(polo em vez de zero).

---

## 4. Casos de teste — soluções de referência

> **Tolerâncias:** valores numéricos batem em ~1e-3 (relativo).
> Os polos da malha fechada devem cair em `sd` com erro < 1e-10.

### 4.0 Validação cruzada com a apostila (rodar PRIMEIRO)

| Apostila | Topologia | Esperado | Calculado |
|---|---|---|---|
| §4.2 — `G=2/s²`, `sd=-1+1.95j` | PD | `z=2.41`, `Kt=2` | `z=2.401`, `Kt=2.000` |
| §4.4 — `G=1/(s²+1)`, `sd=-1+√3·j` | PID `z₁=z₂` | `z=0.79`, `Kc=2.37` | `z=0.789`, `Kc=2.369` |
| §4.5 — `G=4/[s(s+2)]`, `sd=-2+3.464j`, `z=3` fixo | Lead | `p=5.6` | `p=5.600` |

### 4.1 — Exercício 1 (PD)

**Planta**: `G(s) = 4(s+4)/(s³+4s²+4s) = 4(s+4)/[s(s+2)²]`
**Realimentação**: `H(s) = 1`
**Especs**: `Mp ≤ 10%`, `ts(5%) < 4s`

| Grandeza | Valor |
|---|---|
| ζ | 0.5912 |
| ωn,min | 1.2687 rad/s |
| sd | `-0.7500 + 1.0233j` |
| `G(sd)·H(sd)` | módulo 4.117, ângulo +172.63° |
| θ_c | +7.371° |
| **z** | **8.6598** |
| **Kc = Kd** | **0.030458** |
| **Kp = Kc·z** | **0.26376** |

Polos MF: `-2.622`, `-0.75 ± 1.023j` (sd exato).
Mp ≈ 9.18%, ts(5%) ≈ 4.13s — atende Mp; ts fica no limite (use multiplicador
de Kc < 1 ou aumente ωn no modo manual).

### 4.2 — Exercício 2 (PD com planta instável)

**Planta**: `G(s) = 1/[10000·(s² − 1.1772)]` (polos em ±1.085)
**Realimentação**: `H(s) = 1`
**Especs**: `ζ = 0.7`, `ωn = 0.5 rad/s`

| Grandeza | Valor |
|---|---|
| sd | `-0.3500 + 0.3571j` |
| `G(sd)·H(sd)` | módulo 8.276e-5, ângulo +168.06° |
| θ_c | +11.94° |
| **z** | **2.0389** |
| **Kc = Kd** | **7000.0** |
| **Kp = Kc·z** | **14272.0** |

Polos MF: `-0.35 ± 0.357j` (sd exato).

### 4.3 — Exercício 3 (PI)

**Planta**: `G(s) = 5(s²+5s+4)/(s²+4s+4) = 5(s+1)(s+4)/(s+2)²`
**Realimentação**: `H(s) = 0.2/(s+1)`
**sd dado**: `-4 + 4j`

| Grandeza | Valor |
|---|---|
| `G(sd)·H(sd)` | módulo 0.20, ângulo −143.13° |
| θ_c | +98.13° |
| **z** | **3.4286** (= 24/7) |
| **Kc = Kp** | **7.000** |
| **Ki = Kc·z** | **24.000** |

Polos MF: `-4 ± 4j`, `-3`, `-1`.
Em regime, `y(∞) = 1/H(0) = 5` para entrada degrau unitário.

### 4.4 — Exercício 4 (PID, z₁ = z₂)

**Planta**: `G(s) = 5/(s³+12s²+22s+20)` (polos em −10, −1±j)
**Realimentação**: `H(s) = 0.4`
**Especs**: `Mp ≤ 20%`, `ts(2%) < 5s`

| Grandeza | Valor |
|---|---|
| ζ | 0.4559 |
| ωn,min | 1.7546 rad/s |
| sd | `-0.800 + 1.5616j` |
| `G(sd)·H(sd)` | módulo 0.1399, ângulo −165.57° |
| θ_c (cada zero) | +51.346° |
| **z₁ = z₂** | **2.0490** |
| **Kc = Kd** | **3.1360** |
| **Kp = 2·Kc·z** | **12.851** |
| **Ki = Kc·z²** | **13.166** |

Polos MF: `-9.500`, `-0.900`, `-0.8 ± 1.562j`.
Mp ≈ 5.4 % ✓, ts(2%) ≈ 4.64 s ✓.

### 4.5 — Exercício 5(a) (PID, z₁ = z₂)

**Planta**: `G(s) = 5(s+3)/[s(s+4)]`
**Realimentação**: `H(s) = 1/(s+1)`
**Especs**: `Mp ≤ 10%`, `ts(5%) < 3s`

| Grandeza | Valor |
|---|---|
| ζ | 0.5912 |
| ωn,min | 1.6916 rad/s |
| sd | `-1.000 + 1.3644j` |
| `G(sd)·H(sd)` | módulo 1.5915, ângulo +153.61° |
| θ_c | +76.316° |
| **z₁ = z₂** | **1.3322** |
| **Kc = Kd** | **0.5390** |
| **Kp = 2·Kc·z** | **1.4362** |
| **Ki = Kc·z²** | **0.9567** |

Polos MF: `-4.607`, `-1.089`, `-1 ± 1.364j`.

### 4.6 — Exercício 5(b) — Tustin com T = 2 s

A partir de `Gc(s) = 0.5390·(s+1.3322)²/s`:

```
        2.9319·z² + 0.8352·z + 0.0595
Gc(z) = ─────────────────────────────
                 z² − 1
```

### 4.7 — Exercício 6(a) — Compensador `a/(s+b)`

**Planta**: `G(s) = 2(s+1)/(s²+2s+2)`
**Realimentação**: `H(s) = (s+3)/(s+5)`
**sd dado**: `-2.5 + 2j`

| Grandeza | Valor |
|---|---|
| `G(sd)·H(sd)` | módulo 0.5325, ângulo −98.70° |
| ∠(sd+b) alvo | +81.30° |
| **a** | **3.7999** |
| **b** | **2.8061** |

Polos MF: `-2.5 ± 2j`, `-3.305`, `-1.502`.

### 4.8 — Exercício 6(b) — Euler com T = 1 s, sobre `G·H·Gc`

`G(s)·H(s)·Gc(s)` em s:
- num: `7.5998·s² + 30.399·s + 22.799`
- den: `s⁴ + 9.806·s³ + 31.643·s² + 43.673·s + 28.061`

Após Euler `s ← z−1`, normalizado (denominador mônico):

```
              7.600·z² + 15.200·z
G·H·Gc(z) = ─────────────────────────────────────────────
            z⁴ + 5.806·z³ + 8.224·z² + 5.806·z + 7.224
```

### 4.9 — Arquivo de testes

Crie `tests/test_lista.py`:

```python
import numpy as np
import pytest
from lgr import (zeta_de_mp, wn_min, polo_desejado,
                 projeto_PD, projeto_PI, projeto_PID_zigual,
                 projeto_compensador_apb,
                 malha_fechada, c2d_tustin, c2d_euler)

# --- Apostila ----------------------------------------------------------
def test_apostila_PD():
    sd = -1 + 1.95j
    r = projeto_PD([2.0], [1.0, 0, 0], [1.0], [1.0], sd)
    assert abs(r["z"] - 2.41) < 0.02
    # Kc deste código = Kt da apostila
    assert abs(r["Kc"] - 2.0) < 0.05

def test_apostila_PID_zigual():
    sd = -1 + np.sqrt(3) * 1j
    r = projeto_PID_zigual([1.0], [1.0, 0, 1.0], [1.0], [1.0], sd)
    assert abs(r["z1"] - 0.79) < 0.02
    assert abs(r["Kc"] - 2.37) < 0.02

# --- Lista DCA-3701 ----------------------------------------------------
def test_ex1_PD():
    zeta = zeta_de_mp(10)
    wn = wn_min(zeta, 4.0, "5%")
    sd = polo_desejado(zeta, wn)
    r = projeto_PD([4.0, 16.0], [1.0, 4.0, 4.0, 0.0], [1.0], [1.0], sd)
    assert abs(r["z"] - 8.6598) < 1e-3
    assert abs(r["Kc"] - 0.030458) < 1e-5
    # Polo MF deve cair em sd
    nT, dT = malha_fechada([4.0, 16.0], [1.0, 4.0, 4.0, 0.0],
                           [1.0], [1.0], r["Gc_num"], r["Gc_den"])
    polos = np.roots(dT)
    d = np.min(np.abs(polos - sd))
    assert d < 1e-9

def test_ex2_PD():
    sd = polo_desejado(0.7, 0.5)
    r = projeto_PD([1.0], [10000.0, 0.0, -10000*1.1772], [1.0], [1.0], sd)
    assert abs(r["z"] - 2.0389) < 1e-3
    assert abs(r["Kc"] - 7000.0) < 1.0

def test_ex3_PI():
    sd = -4 + 4j
    r = projeto_PI([5.0, 25.0, 20.0], [1.0, 4.0, 4.0],
                   [0.2], [1.0, 1.0], sd)
    assert abs(r["z"] - 24/7) < 1e-3
    assert abs(r["Kc"] - 7.0) < 1e-3
    assert abs(r["Ki"] - 24.0) < 1e-3

def test_ex4_PID():
    zeta = zeta_de_mp(20)
    wn = wn_min(zeta, 5.0, "2%")
    sd = polo_desejado(zeta, wn)
    r = projeto_PID_zigual([5.0], [1.0, 12.0, 22.0, 20.0],
                           [0.4], [1.0], sd)
    assert abs(r["z1"] - 2.0490) < 1e-3
    assert abs(r["Kc"] - 3.1360) < 1e-3

def test_ex5a_PID():
    zeta = zeta_de_mp(10)
    wn = wn_min(zeta, 3.0, "5%")
    sd = polo_desejado(zeta, wn)
    r = projeto_PID_zigual([5.0, 15.0], [1.0, 4.0, 0.0],
                           [1.0], [1.0, 1.0], sd)
    assert abs(r["z1"] - 1.3322) < 1e-3
    assert abs(r["Kc"] - 0.5390) < 1e-3

def test_ex5b_tustin():
    Kc, z = 0.5390, 1.3322
    num_Gc = Kc * np.array([1.0, 2*z, z*z])
    den_Gc = np.array([1.0, 0.0])
    nz, dz = c2d_tustin(num_Gc, den_Gc, 2.0)
    np.testing.assert_allclose(nz, [2.9319, 0.8352, 0.0595], atol=1e-3)
    np.testing.assert_allclose(dz, [1.0, 0.0, -1.0], atol=1e-9)

def test_ex6a_apb():
    sd = -2.5 + 2.0j
    r = projeto_compensador_apb([2.0, 2.0], [1.0, 2.0, 2.0],
                                 [1.0, 3.0], [1.0, 5.0], sd)
    assert abs(r["a"] - 3.7999) < 1e-3
    assert abs(r["b"] - 2.8061) < 1e-3

def test_ex6b_euler():
    # Pega Gc do (a) e discretiza G·H·Gc com T=1
    sd = -2.5 + 2.0j
    r = projeto_compensador_apb([2.0, 2.0], [1.0, 2.0, 2.0],
                                 [1.0, 3.0], [1.0, 5.0], sd)
    from lgr import tf_mul
    n_GH, d_GH = tf_mul([2.0, 2.0], [1.0, 2.0, 2.0],
                        [1.0, 3.0], [1.0, 5.0])
    n_tot, d_tot = tf_mul(n_GH, d_GH, r["Gc_num"], r["Gc_den"])
    nz, dz = c2d_euler(n_tot, d_tot, 1.0)
    np.testing.assert_allclose(
        dz, [1.0, 5.806, 8.224, 5.806, 7.224], atol=1e-3
    )
```

---

## 5. Presets — `presets.py`

Dicionário com as 6 questões + um preset "vazio" para entrada genérica:

```python
PRESETS = {
    "— (entrada manual) —": None,

    "Ex 1 — PD (Mp≤10%, ts5%<4s)": {
        "descricao": "G(s) = 4(s+4)/(s³+4s²+4s), H(s)=1. PD para Mp≤10%, ts(5%)<4s.",
        "numG": [4, 16], "denG": [1, 4, 4, 0],
        "numH": [1],     "denH": [1],
        "topologia": "PD",
        "modo_specs": "mp_ts",
        "Mp": 10.0, "ts": 4.0, "criterio_ts": "5%",
    },

    "Ex 2 — PD (ζ=0.7, ωn=0.5)": {
        "descricao": "G(s)=1/[10000(s²-1.1772)] (instável), H(s)=1. PD para ζ=0.7, ωn=0.5.",
        "numG": [1], "denG": [10000, 0, -11772],
        "numH": [1], "denH": [1],
        "topologia": "PD",
        "modo_specs": "zeta_wn",
        "zeta": 0.7, "wn": 0.5,
    },

    "Ex 3 — PI (sd = -4 ± 4j)": {
        "descricao": "G=5(s²+5s+4)/(s²+4s+4), H=0.2/(s+1). PI com polos em -4±4j.",
        "numG": [5, 25, 20], "denG": [1, 4, 4],
        "numH": [0.2],       "denH": [1, 1],
        "topologia": "PI",
        "modo_specs": "polos",
        "sd_re": -4.0, "sd_im": 4.0,
    },

    "Ex 4 — PID z₁=z₂ (Mp≤20%, ts2%<5s)": {
        "descricao": "G=5/(s³+12s²+22s+20), H=0.4. PID com z₁=z₂ para Mp≤20%, ts(2%)<5s.",
        "numG": [5], "denG": [1, 12, 22, 20],
        "numH": [0.4], "denH": [1],
        "topologia": "PID_zigual",
        "modo_specs": "mp_ts",
        "Mp": 20.0, "ts": 5.0, "criterio_ts": "2%",
    },

    "Ex 5 — PID z₁=z₂ + Tustin": {
        "descricao": "G=5(s+3)/[s(s+4)], H=1/(s+1). PID z₁=z₂ para Mp≤10%, ts(5%)<3s; depois Tustin T=2s.",
        "numG": [5, 15], "denG": [1, 4, 0],
        "numH": [1],     "denH": [1, 1],
        "topologia": "PID_zigual",
        "modo_specs": "mp_ts",
        "Mp": 10.0, "ts": 3.0, "criterio_ts": "5%",
        "discretizar": True, "metodo_disc": "tustin", "T": 2.0,
        "discretizar_alvo": "Gc",
    },

    "Ex 6 — a/(s+b) + Euler": {
        "descricao": "G=2(s+1)/(s²+2s+2), H=(s+3)/(s+5). Compensador a/(s+b) para sd=-2.5±2j; depois Euler T=1s sobre G·H·Gc.",
        "numG": [2, 2], "denG": [1, 2, 2],
        "numH": [1, 3], "denH": [1, 5],
        "topologia": "apb",
        "modo_specs": "polos",
        "sd_re": -2.5, "sd_im": 2.0,
        "discretizar": True, "metodo_disc": "euler", "T": 1.0,
        "discretizar_alvo": "GHGc",
    },
}
```

---

## 6. Interface Streamlit — `app.py`

### 6.1 Layout geral

```python
import streamlit as st
from presets import PRESETS

st.set_page_config(page_title="Projeto LGR", layout="wide")
st.title("Projeto de Controladores via Lugar das Raízes")
st.caption("DCA-3701 — UFRN. Suporta P, PD, PI, PID e a/(s+b).")

# Sidebar: seleção de preset e edição manual
with st.sidebar:
    nome = st.selectbox("Carregar exemplo (preset)", list(PRESETS.keys()))
    preset = PRESETS[nome]
    if preset:
        st.info(preset["descricao"])
        if st.button("Aplicar preset"):
            st.session_state.update(preset)
            st.rerun()
```

### 6.2 Entrada — uma única "página" (sem multi-page)

Use `st.expander` para colapsar seções. Estrutura:

```
[Sidebar]
  Preset selector

[Main]
  1. Planta e realimentação
     ├─ Numerador G(s)    [text_input "4 4" → [4, 4]]
     ├─ Denominador G(s)
     ├─ Numerador H(s)
     └─ Denominador H(s)

  2. Tipo de controlador
     └─ selectbox: P / PD / PI / PID (z₁=z₂) / a/(s+b)

  3. Especificações de desempenho
     ├─ radio: "Mp + ts" | "ζ + ωn" | "Polos sd diretamente"
     └─ inputs específicos do modo

  4. Discretização (opcional, expander)
     ├─ checkbox "Discretizar"
     ├─ radio: Tustin / Euler
     ├─ number_input: T (s)
     └─ radio: discretizar Gc(s) ou G(s)·H(s)·Gc(s)

  5. Multiplicador de Kc (slider 0.5 a 1.5, default 1.0)
     [tooltip: "Ajuste fino. Reduza se Mp da simulação estourar."]

  [Botão] PROJETAR

  --- Resultados (após clicar) ---

  Passo 1 — Especificações e sd
  Passo 2 — G(sd)·H(sd)
  Passo 3 — Equação angular
  Passo 4 — Posição do(s) zero(s)/polo(s)
  Passo 5 — Ganho Kc
  Passo 6 — Equação final do controlador
  Passo 7 — Validação (3 gráficos lado a lado)
  Passo 8 — Tabela "obtido vs especificado"
  Passo 9 — Discretização [se ativada]
```

### 6.3 Apresentação de cada passo

Use `st.latex(r"...")` para fórmulas e `st.markdown(r"$...$")` inline.
Cada passo dentro de `st.expander(..., expanded=True)` para o usuário poder
colapsar.

**Exemplo de layout do Passo 1:**

```python
with st.expander("Passo 1 — Especificações e polo dominante", expanded=True):
    st.latex(r"\zeta = \frac{-\ln(M_p/100)}{\sqrt{\pi^2 + \ln^2(M_p/100)}}")
    st.markdown(f"$M_p = {Mp}\\%$  ⇒  $\\zeta = {zeta:.4f}$")
    st.latex(r"\omega_n \geq \frac{k}{\zeta\,t_s}")
    st.markdown(f"$t_s({criterio}) \\leq {ts}\\,s$  ⇒  $\\omega_n \\geq {wn_min:.4f}$ rad/s")
    st.latex(r"s_d = -\zeta\omega_n + j\omega_n\sqrt{1-\zeta^2}")
    st.markdown(f"$s_d = {sd.real:.4f} + {sd.imag:.4f}\\,j$")
```

Fórmula análoga para os outros passos (veja a pedagogia da apostila).

### 6.4 Gráficos do Passo 7

Três gráficos lado a lado em `st.columns(3)`:

1. **Resposta ao degrau de MF** — com linhas tracejadas em `yss·(1±tol)`
   (tol = 0.02 ou 0.05 conforme o critério). Anotar `Mp` e `ts` com `text`.
2. **Mapa polos/zeros de MF** — `marker='x'` para polos, `marker='o'` para
   zeros. Plotar também `sd` e seu conjugado em outra cor.
3. **LGR esboçado** — varrer `K = np.logspace(-2, 2, 200)` e plotar raízes
   de `1 + K·numL/denL = 0`. Marcar o ganho de projeto.

**Importante**: tamanhos limitados (~ 4 in × 3 in) e `dpi=100` para evitar
demora na renderização.

### 6.5 Tabela do Passo 8

```python
import pandas as pd
df = pd.DataFrame({
    "Grandeza":     ["ζ", "ωn (rad/s)", "Mp (%)", f"ts({crit}) (s)", "polo MF mais perto de sd"],
    "Obtido":       [zeta_obt, wn_obt, Mp_obt, ts_obt, polo_perto],
    "Especificado": [zeta_esp, wn_esp, Mp_esp, ts_esp, sd],
})
st.dataframe(df, use_container_width=True, hide_index=True)
```

### 6.6 Passo 9 — Discretização

Se ativada, mostre:
1. `Gc(z)` (ou `G·H·Gc(z)`) em formato polinomial.
2. Equação de diferenças no domínio do tempo:
   ```
   u[k] = -a₁·u[k-1] - a₂·u[k-2] - ... + b₀·e[k] + b₁·e[k-1] + ...
   ```

---

## 7. Estrutura de arquivos

```
.
├── app.py                  # entrada Streamlit
├── lgr.py                  # núcleo numérico (§3)
├── presets.py              # presets dos 6 exercícios (§5)
├── plotting.py             # funções de gráfico (§6.4)
├── tests/
│   ├── test_apostila.py    # casos da §4.0
│   └── test_lista.py       # 6 exercícios (§4.9)
├── requirements.txt
└── README.md
```

`requirements.txt`:
```
streamlit>=1.30
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
pytest>=7.4
```

---

## 8. Ordem de implementação (siga estritamente)

1. **`lgr.py` primeiro, sem Streamlit**. Implemente todas as funções e
   rode os testes da §4 com `print()`. Garanta que **os polos MF batem em
   sd até 1e-10** para os 6 exercícios.
2. **`tests/test_lista.py`** com pytest. Rode `pytest tests/` e garanta
   que tudo passa antes de tocar na UI.
3. **`plotting.py`**: três funções de gráfico. Teste fora do Streamlit
   primeiro (salvando em PNG e abrindo manualmente).
4. **`presets.py`**: dicionário simples.
5. **`app.py`** Streamlit. Comece com hardcoding o Ex 1 inteiro,
   depois generalize.
6. Adicione discretização ao final (Passo 9).

> **Critério de "pronto":** rodar `pytest tests/` deve passar em todos os
> casos antes de marcar o app como completo.

---

## 9. Armadilhas conhecidas (releia antes de codar)

- **Ramo do ângulo (±180°).** A escolha entre `+180°` e `−180°` na
  equação angular importa. Sempre teste se o ângulo resultante para o
  zero está em `(0°, 180°)`. Se não, troque o sinal e refaça.
- **Avaliação de G(s) imaginária.** Use `np.polyval(num, s) / np.polyval(den, s)`.
  Coeficientes em **ordem decrescente** (convenção numpy). Erros aqui
  produzem ângulos invertidos e zeros no semiplano direito (bug clássico).
- **`scipy.signal.cont2discrete` não aceita TFs impróprias.** Para PID
  puro `(s+z)²/s`, use a substituição polinomial direta da §2.7.
- **Mp inflado.** Sempre meça Mp DEPOIS do primeiro cruzamento de `yss`
  (§2.6). Sem isso, no Ex 3 e 5 o "Mp" sai gigante.
- **`yss ≠ 1`.** Quando `H(s)` não tem ganho DC = 1, a saída em regime
  ao degrau não é 1. Mostre `yss` no painel e calcule Mp/ts relativos a
  `yss`. No Ex 3, `yss = 5`. No Ex 6, `yss ≈ 0.747`.
- **Ex 1 e 5a no limite.** Com `ωn = ωn,min`, ts fica exatamente no limite
  e a simulação numérica pode mostrá-lo ligeiramente acima (efeito dos zeros
  do controlador). Solução: o usuário pode reduzir o **multiplicador de Kc**
  (~ 0.9) para baixar Mp, ou trocar para o modo `ζ + ωn` e digitar um `ωn`
  maior. Os polos MF caem exatamente em sd, então o método está correto —
  o desvio é da aproximação de 2ª ordem.
- **Sinal de `z` negativo.** Se a fórmula retorna `z < 0`, o zero está no
  SPD (fase não-mínima). Avise o usuário com `st.warning`.
- **PID puro é impróprio.** Tem 2 zeros e 1 polo. Não tente convertê-lo
  para state-space — use a substituição direta da §2.7.
- **Coeficientes dos polinômios.** Sempre `float`, nunca `int`. `[1, 4, 4]`
  é OK (numpy converte), mas `[10000, 0, -10000*1.1772]` precisa virar
  `[10000.0, 0.0, -11772.0]` para evitar `TypeError` em alguns paths do scipy.
- **Não use `python-control`.** Já dito, mas vale repetir — quebra a
  pedagogia "à mão" do app.

---

## 10. Critérios de aceitação

✅ `pytest tests/` passa nos 11 testes (3 da apostila + 8 da lista).
✅ Para cada um dos 6 presets, ao clicar "Projetar":
   - os passos 1-9 aparecem todos;
   - os gráficos renderizam sem erro;
   - a tabela do passo 8 mostra valores compatíveis com a §4.
✅ Em modo "entrada manual", o usuário consegue inserir uma planta
   arbitrária e obter o controlador.
✅ O ajuste "multiplicador de Kc" altera o resultado visualmente
   (a curva muda).
✅ Discretização (Tustin/Euler) funciona e produz `Gc(z)` que bate com a
   §4.6 (Ex 5b) e §4.8 (Ex 6b).
