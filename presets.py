"""Presets dos 6 exercícios da lista DCA-3701 (§5 do CLAUDE.md)."""

PRESETS = {
    "— (entrada manual) —": None,

    "Ex 1 — PD (Mp≤10%, ts5%<4s)": {
        "descricao": (
            "G(s) = 4(s+4)/(s³+4s²+4s), H(s)=1. "
            "Projetar PD para Mp≤10%, ts(5%)<4s."
        ),
        "numG": [4, 16],
        "denG": [1, 4, 4, 0],
        "numH": [1],
        "denH": [1],
        "topologia": "PD",
        "modo_specs": "mp_ts",
        "Mp": 10.0,
        "ts": 4.0,
        "criterio_ts": "5%",
    },

    "Ex 2 — PD (ζ=0.7, ωn=0.5)": {
        "descricao": (
            "G(s) = 1/[10000(s²−1.1772)] (planta instável), H(s)=1. "
            "Projetar PD para ζ=0.7, ωn=0.5."
        ),
        "numG": [1],
        "denG": [10000, 0, -11772],
        "numH": [1],
        "denH": [1],
        "topologia": "PD",
        "modo_specs": "zeta_wn",
        "zeta": 0.7,
        "wn": 0.5,
    },

    "Ex 3 — PI (sd = -4 ± 4j)": {
        "descricao": (
            "G(s) = 5(s²+5s+4)/(s²+4s+4), H(s) = 0.2/(s+1). "
            "PI com polos em −4 ± 4j."
        ),
        "numG": [5, 25, 20],
        "denG": [1, 4, 4],
        "numH": [0.2],
        "denH": [1, 1],
        "topologia": "PI",
        "modo_specs": "polos",
        "sd_re": -4.0,
        "sd_im": 4.0,
    },

    "Ex 4 — PID z₁=z₂ (Mp≤20%, ts2%<5s)": {
        "descricao": (
            "G(s) = 5/(s³+12s²+22s+20), H(s) = 0.4. "
            "PID com z₁=z₂ para Mp≤20%, ts(2%)<5s."
        ),
        "numG": [5],
        "denG": [1, 12, 22, 20],
        "numH": [0.4],
        "denH": [1],
        "topologia": "PID_zigual",
        "modo_specs": "mp_ts",
        "Mp": 20.0,
        "ts": 5.0,
        "criterio_ts": "2%",
    },

    "Ex 5 — PID z₁=z₂ + Tustin": {
        "descricao": (
            "G(s) = 5(s+3)/[s(s+4)], H(s) = 1/(s+1). "
            "PID z₁=z₂ para Mp≤10%, ts(5%)<3s; depois Tustin com T=2s."
        ),
        "numG": [5, 15],
        "denG": [1, 4, 0],
        "numH": [1],
        "denH": [1, 1],
        "topologia": "PID_zigual",
        "modo_specs": "mp_ts",
        "Mp": 10.0,
        "ts": 3.0,
        "criterio_ts": "5%",
        "discretizar": True,
        "metodo_disc": "tustin",
        "T": 2.0,
        "discretizar_alvo": "Gc",
    },

    "Ex 6 — a/(s+b) + Euler": {
        "descricao": (
            "G(s) = 2(s+1)/(s²+2s+2), H(s) = (s+3)/(s+5). "
            "Compensador a/(s+b) para sd=-2.5±2j; depois Euler com T=1s "
            "sobre G·H·Gc."
        ),
        "numG": [2, 2],
        "denG": [1, 2, 2],
        "numH": [1, 3],
        "denH": [1, 5],
        "topologia": "apb",
        "modo_specs": "polos",
        "sd_re": -2.5,
        "sd_im": 2.0,
        "discretizar": True,
        "metodo_disc": "euler",
        "T": 1.0,
        "discretizar_alvo": "GHGc",
    },
}
