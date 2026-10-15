"""
MODE (Method for Object-based Diagnostic Evaluation) em Python.

Reproduz, para campos 2D numa grade regular, o MODE do MET v12.2.1 (DTC/NCAR): mesmos objetos, atributos,
interesse total e MMI (validado contra o executável `mode` do MET; ver README).

Convenção da grade: matriz [linha, coluna] = [y, x], com a linha 0 ao SUL (latitude crescente), como o MET lê
uma grade lat/lon regular. Distâncias são medidas em pontos de grade × grid_res_km (grade tratada como uniforme,
igual ao MET).

Uso mínimo:
    from mode_python import ModeConfig, avaliar
    r = avaliar(obs, prev, ModeConfig())
    r.mmi, r.pares, r.clusters
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve, label

# ======================================================================================
# CONFIGURAÇÃO
# ======================================================================================


@dataclass
class ModeConfig:
    """Parâmetros do MODE. Os padrões são os do MODEConfig_default do MET v12.2.1."""

    grid_res_km: float = 60.0                 # tamanho do ponto de grade (km)
    conv_radius: int = 1                      # raio do disco de convolução (pontos de grade); 0 = sem convolução
    conv_thresh: float = 5.0                  # limiar (>=) no campo convolvido
    vld_thresh: float = 0.5                   # fração mínima de dados válidos no disco (centro ausente)
    min_area: int = 0                         # descarta objetos com área < min_area pontos (0 = não filtra)
    merge_thresh: float | None = 1.25         # limiar baixo para agrupar objetos em clusters (None = sem merging)
    max_centroid_dist_km: float = 800.0       # pares mais distantes não são avaliados
    total_interest_thresh: float = 0.70       # interesse mínimo para formar um par

    # pesos do interesse total
    weight: dict = field(default_factory=lambda: {
        "centroid_dist": 2.0, "boundary_dist": 4.0, "angle_diff": 1.0, "area_ratio": 1.0, "int_area_ratio": 2.0})

    # funções de interesse: pontos (x, interesse), lineares por partes, constantes fora do intervalo
    # distâncias em km, ângulo em graus, razões adimensionais
    interest: dict = field(default_factory=lambda: {
        "centroid_dist": [(0.0, 1.0), (60.0, 1.0), (600.0, 0.0)],
        "boundary_dist": [(0.0, 1.0), (400.0, 0.0)],
        "angle_diff": [(0.0, 1.0), (30.0, 1.0), (90.0, 0.0)],
        "area_ratio": [(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)],
        "int_area_ratio": [(0.0, 0.0), (0.10, 0.5), (0.25, 1.0), (1.0, 1.0)]})

    @classmethod
    def dissertacao(cls):
        """Configuração usada na dissertação (chuva 24 h, grade de 0,6° ≈ 60 km)."""
        return cls(conv_radius=3, conv_thresh=10.0, min_area=5, merge_thresh=3.0, max_centroid_dist_km=1100.0,
                   total_interest_thresh=0.55,
                   weight={"centroid_dist": 3.0, "boundary_dist": 2.5, "angle_diff": 1.0, "area_ratio": 0.5,
                           "int_area_ratio": 1.5},
                   interest={"centroid_dist": [(0, 1), (100, 0.9), (200, 0.7), (400, 0.3), (600, 0)],
                             "boundary_dist": [(0, 1), (60, 0.8), (180, 0.4), (400, 0)],
                             "angle_diff": [(0, 1), (30, 0.5), (90, 0)],
                             "area_ratio": [(0, 0), (0.8, 1), (1, 1)],
                             "int_area_ratio": [(0, 0), (0.2, 0.5), (0.5, 0.8), (1, 1)]})


# ======================================================================================
# 1. OBJETOS
# ======================================================================================

def convolver(campo, raio, vld_thresh=0.5):
    """Média num disco de raio `raio` (pontos com distância <= raio), só com pontos válidos DENTRO da grade."""
    campo = np.asarray(campo, dtype=float)
    if raio <= 0:
        return campo.copy()
    y, x = np.mgrid[-raio:raio + 1, -raio:raio + 1]
    disco = (x ** 2 + y ** 2 <= raio ** 2).astype(float)
    valido = np.isfinite(campo)
    soma = convolve(np.where(valido, campo, 0.0), disco, mode="constant", cval=0.0)
    n_vld = convolve(valido.astype(float), disco, mode="constant", cval=0.0)
    n_tot = convolve(np.ones_like(campo), disco, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = soma / n_vld
    c[n_vld == 0] = np.nan
    c[~valido & (n_vld / n_tot < vld_thresh)] = np.nan
    return c


def identificar(campo, cfg):
    """Convolução + limiar + objetos conexos (8 vizinhos) + filtro de área. Retorna (rótulos, campo convolvido)."""
    conv = convolver(campo, cfg.conv_radius, cfg.vld_thresh)
    with np.errstate(invalid="ignore"):
        mascara = conv >= cfg.conv_thresh
    rot, n = label(mascara, structure=np.ones((3, 3)))
    if cfg.min_area > 0:
        areas = np.bincount(rot.ravel())
        pequenos = np.nonzero(areas < cfg.min_area)[0]
        rot[np.isin(rot, pequenos[pequenos > 0])] = 0
        rot, n = label(rot > 0, structure=np.ones((3, 3)))
    return rot, conv


def _vertices(m):
    """Vértices dos pixels do objeto: ponto (x, y) ligado se (x,y), (x−1,y), (x,y−1) ou (x−1,y−1) está no objeto."""
    f = m.copy()
    f[:, 1:] |= m[:, :-1]
    f[1:, :] |= m[:-1, :]
    f[1:, 1:] |= m[:-1, :-1]
    return f


def atributos(mascara, campo):
    """Atributos de um objeto. Centroide, ângulo, comprimento e largura usam os vértices; a área, os pixels."""
    fv = _vertices(mascara)
    yy, xx = np.nonzero(fv)
    x, y = xx.astype(float), yy.astype(float)
    xb, yb = x.mean(), y.mean()
    ang = 0.5 * np.arctan2(2 * ((x - xb) * (y - yb)).sum(), ((x - xb) ** 2).sum() - ((y - yb) ** 2).sum())
    u = x * np.cos(ang) + y * np.sin(ang)
    v = x * np.cos(ang + np.pi / 2) + y * np.sin(ang + np.pi / 2)
    comprimento, largura = sorted((u.max() - u.min(), v.max() - v.min()), reverse=True)
    p = np.pad(fv, 1)
    borda = fv & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    chuva = np.asarray(campo, dtype=float)[mascara]
    return {"area": int(mascara.sum()), "centroide_x": xb, "centroide_y": yb, "angulo": np.degrees(ang),
            "comprimento": comprimento, "largura": largura,
            "razao_aspecto": largura / comprimento if comprimento > 0 else 1.0,
            "p50": float(np.nanpercentile(chuva, 50)), "p90": float(np.nanpercentile(chuva, 90)),
            "soma": float(np.nansum(chuva)), "borda": np.argwhere(borda).astype(float), "mascara": mascara}


# ======================================================================================
# 2. INTERESSE DE CADA PAR
# ======================================================================================

def _pwl(x, pontos):
    xs, ys = zip(*sorted(pontos))
    return float(np.interp(x, xs, ys))


def _conf_aspecto(r):
    return ((r - 1.0) ** 2 / (r ** 2 + 1.0)) ** 0.3


def interesse_par(o, f, cfg):
    """Interesse total de um par (obs, prev) e seus componentes. Retorna None se os centroides passam do limite."""
    g = cfg.grid_res_km
    d_cent = np.hypot(o["centroide_x"] - f["centroide_x"], o["centroide_y"] - f["centroide_y"]) * g
    if d_cent > cfg.max_centroid_dist_km:
        return None
    inter = int(np.logical_and(o["mascara"], f["mascara"]).sum())
    if inter > 0:
        d_borda = 0.0                                         # sobrepostos ou um dentro do outro
    else:
        a, b = o["borda"], f["borda"]
        d_borda = float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min()) * g
    dang = abs(o["angulo"] - f["angulo"]) % 180.0
    dang = 180.0 - dang if dang > 90.0 else dang
    razao_area = min(o["area"], f["area"]) / max(o["area"], f["area"])
    razao_inter = inter / min(o["area"], f["area"])

    comp = {"centroid_dist": (d_cent, razao_area),          # (atributo, confiança)
            "boundary_dist": (d_borda, 1.0),
            "angle_diff": (dang, np.sqrt(_conf_aspecto(o["razao_aspecto"]) * _conf_aspecto(f["razao_aspecto"]))),
            "area_ratio": (razao_area, 1.0),
            "int_area_ratio": (razao_inter, 1.0)}
    num = den = 0.0
    for k, (valor, conf) in comp.items():
        w = cfg.weight.get(k, 0.0)
        num += w * conf * _pwl(valor, cfg.interest[k])
        den += w * conf
    return {"interesse": num / den if den > 0 else np.nan, "dist_centroide_km": d_cent, "dist_borda_km": d_borda,
            "dif_angulo": dang, "razao_area": razao_area, "razao_intersecao": razao_inter}


# ======================================================================================
# 3. RESULTADO
# ======================================================================================

@dataclass
class Resultado:
    obs: list                 # atributos dos objetos observados
    prev: list                # atributos dos objetos previstos
    rotulos_obs: np.ndarray
    rotulos_prev: np.ndarray
    interesse: np.ndarray     # [n_obs, n_prev]; NaN = par não avaliado (centroides além do limite)
    detalhes: dict            # (i_obs, j_prev) -> componentes do interesse
    pares: list               # (i_obs, j_prev, interesse) com interesse >= limiar
    clusters: list            # [{"obs": [...], "prev": [...]}], índices a partir de 0
    mmi: float                # mediana dos interesses máximos (obs + prev)
    mmif: float               # idem, só objetos previstos
    mmio: float               # idem, só objetos observados


def _clusters(pares, grupos_obs, grupos_prev):
    """Une pares e grupos do merging; mantém só conjuntos com objeto observado E previsto (como o MET)."""
    conj = [({i}, {j}) for i, j, _ in pares]
    conj += [(set(g), set()) for g in grupos_obs] + [(set(), set(g)) for g in grupos_prev]
    mudou = True
    while mudou:
        mudou = False
        for a in range(len(conj)):
            for b in range(a + 1, len(conj)):
                if conj[a][0] & conj[b][0] or conj[a][1] & conj[b][1]:
                    conj[a] = (conj[a][0] | conj[b][0], conj[a][1] | conj[b][1])
                    del conj[b]
                    mudou = True
                    break
            if mudou:
                break
    return [{"obs": sorted(o), "prev": sorted(p)} for o, p in conj if o and p]


def _grupos_merging(rot, conv, n, cfg):
    """Objetos contidos no mesmo objeto do limiar baixo (merge_thresh) formam um grupo."""
    if cfg.merge_thresh is None or n == 0:
        return []
    with np.errstate(invalid="ignore"):
        baixo, _ = label(conv >= cfg.merge_thresh, structure=np.ones((3, 3)))
    dono = {}
    for k in range(1, n + 1):
        ids = np.unique(baixo[rot == k])
        if len(ids) != 1 or ids[0] == 0:          # o MET só faz merging se cada objeto cabe em um único objeto baixo
            return []
        dono.setdefault(int(ids[0]), []).append(k - 1)
    return [g for g in dono.values() if len(g) > 1]


def _mmi(maximos):
    return float(np.median(maximos)) if len(maximos) else np.nan


def avaliar(obs, prev, cfg=None):
    """Roda o MODE para um par de campos 2D (mesma grade; linha 0 ao sul)."""
    cfg = cfg or ModeConfig()
    rot_o, conv_o = identificar(obs, cfg)
    rot_f, conv_f = identificar(prev, cfg)
    n_o, n_f = int(rot_o.max()), int(rot_f.max())
    ao = [atributos(rot_o == k, obs) for k in range(1, n_o + 1)]
    af = [atributos(rot_f == k, prev) for k in range(1, n_f + 1)]

    matriz = np.full((n_o, n_f), np.nan)
    detalhes = {}
    for i, o in enumerate(ao):
        for j, f in enumerate(af):
            d = interesse_par(o, f, cfg)
            if d is not None:
                matriz[i, j], detalhes[i, j] = d["interesse"], d
    pares = [(i, j, matriz[i, j]) for i, j in zip(*np.nonzero(matriz >= cfg.total_interest_thresh))]

    m0 = np.nan_to_num(matriz, nan=0.0)                       # objeto sem par avaliado conta como 0
    max_o = m0.max(axis=1) if n_f else np.zeros(n_o)
    max_f = m0.max(axis=0) if n_o else np.zeros(n_f)
    clusters = _clusters(pares, _grupos_merging(rot_o, conv_o, n_o, cfg), _grupos_merging(rot_f, conv_f, n_f, cfg))
    return Resultado(obs=ao, prev=af, rotulos_obs=rot_o, rotulos_prev=rot_f, interesse=matriz, detalhes=detalhes,
                     pares=pares, clusters=clusters, mmi=_mmi(np.concatenate([max_o, max_f])) if n_o and n_f else 0.0,
                     mmif=_mmi(max_f) if n_o and n_f else 0.0, mmio=_mmi(max_o) if n_o and n_f else 0.0)


def gss_objetos(r, fa=0.05):
    """GSS de objetos (Davis et al. 2009, eqs. 2–4): acertos = objetos previstos com par; fa = fração de área."""
    nf, no = len(r.prev), len(r.obs)
    if no == 0:
        return np.nan
    acertos = len({j for _, j, _ in r.pares})
    perdidos = no - len({i for i, _, _ in r.pares})
    d = (1 - fa) * no / fa - nf
    eps = nf * no / (nf + perdidos + d)
    return (acertos - eps) / (nf + perdidos - eps)
