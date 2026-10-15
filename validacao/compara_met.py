"""Compara o mode_python com o executável `mode` do MET num par de campos (obs × previsão).

Escreve os dois campos em NetCDF CF, monta a configuração do MET a partir da ModeConfig, roda o MET e compara
objetos (número e área), interesse de cada par, MMI e clusters.

Uso:
  python validacao/compara_met.py OBS.nc VAR_OBS PREV.nc VAR_PREV --met /caminho/bin/mode \
         --met-config /caminho/share/met/config/MODEConfig_default [--config padrao|dissertacao] [--saida pasta]

Os arquivos de entrada precisam de coordenadas 1D `lat` e `lon` (grade regular); a latitude é ordenada de sul
para norte antes de rodar os dois códigos.
"""
import argparse, glob, os, re, subprocess, sys, tempfile
import numpy as np, pandas as pd, xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mode_python import ModeConfig, avaliar


def le(arq, var):
    with xr.open_dataset(arq) as ds:
        p = ds[var].squeeze(drop=True).load()
    return p.sortby("lat")


def escreve_cf(p, saida):
    ds = xr.Dataset({"precip": (("time", "lat", "lon"), p.values[None].astype("float32"))},
                    coords={"time": [np.datetime64("2000-01-01T00")], "lat": p.lat.values, "lon": p.lon.values})
    ds.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.lon.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds.precip.attrs = {"units": "mm", "_FillValue": np.float32(-9999.0)}
    ds.time.encoding = {"units": "seconds since 1970-01-01 00:00:00", "dtype": "int64"}
    ds.to_netcdf(saida)


def pwl_met(pontos, distancia):
    return " ".join(f"( {x / 1.0 if not distancia else x}{'/grid_res' if distancia else ''}, {y} )" for x, y in pontos)


def config_met(cfg, padrao):
    """MODEConfig do MET equivalente à ModeConfig (só os itens usados pelo mode_python)."""
    t = open(padrao).read()

    def r(a, b):
        nonlocal t
        n = len(re.findall(a, t, flags=re.S))
        assert n == 1, (a, n)
        t = re.sub(a, lambda m: b, t, flags=re.S)

    w, f = cfg.weight, cfg.interest
    r(r"grid_res = [^;]+;", f"grid_res = {cfg.grid_res_km};")
    r(r"fcst = \{\n   field = \{\n      name  = \"APCP\";\n      level = \"A03\";",
      'fcst = {\n   file_type = NETCDF_NCCF;\n   field = {\n      name  = "precip";\n      level = "(0,*,*)";')
    r(r"conv_radius        = [^;]+;", f"conv_radius        = {cfg.conv_radius};")
    r(r"conv_thresh        = [^;]+;", f"conv_thresh        = >={cfg.conv_thresh};")
    r(r"vld_thresh         = [^;]+;", f"vld_thresh         = {cfg.vld_thresh};")
    if cfg.min_area > 0:
        r(r"filter_attr_name   = \[\];", 'filter_attr_name   = ["AREA"];')
        r(r"filter_attr_thresh = \[\];", f"filter_attr_thresh = [>={cfg.min_area}];")
    if cfg.merge_thresh is None:
        r(r"merge_flag         = [^;]+;", "merge_flag         = NONE;")
    else:
        r(r"merge_thresh       = [^;]+;", f"merge_thresh       = >={cfg.merge_thresh};")
    r(r"max_centroid_dist = [^;]+;", f"max_centroid_dist = {cfg.max_centroid_dist_km}/grid_res;")
    r(r"total_interest_thresh = [^;]+;", f"total_interest_thresh = {cfg.total_interest_thresh};")
    r(r"ps_plot_flag    = [^;]+;", "ps_plot_flag    = FALSE;")
    r(r"weight = \{.*?\n\}", "weight = {\n" + "".join(
        f"   {k} = {w.get(k, 0.0)};\n" for k in ("centroid_dist", "boundary_dist", "convex_hull_dist", "angle_diff",
                                              "aspect_diff", "area_ratio", "int_area_ratio", "curvature_ratio",
                                              "complexity_ratio", "inten_perc_ratio")) + "   inten_perc_value = 50;\n}")
    r(r"interest_function = \{.*?\n\}", "interest_function = {\n"
      f"   centroid_dist = ( {pwl_met(f['centroid_dist'], True)} );\n"
      f"   boundary_dist = ( {pwl_met(f['boundary_dist'], True)} );\n"
      "   convex_hull_dist = ( ( 0.0, 1.0 ) ( 400.0/grid_res, 0.0 ) );\n"
      f"   angle_diff = ( {pwl_met(f['angle_diff'], False)} );\n"
      "   aspect_diff = ( ( 0.00, 1.0 ) ( 0.10, 1.0 ) ( 0.75, 0.0 ) );\n"
      f"   area_ratio = ( {pwl_met(f['area_ratio'], False)} );\n"
      f"   int_area_ratio = ( {pwl_met(f['int_area_ratio'], False)} );\n"
      "   ratio_if = ( ( 0.0, 0.0 ) ( 0.8, 1.0 ) ( 1.0, 1.0 ) );\n"
      "   curvature_ratio = ratio_if;\n   complexity_ratio = ratio_if;\n   inten_perc_ratio = ratio_if;\n}")
    return t


def compara(r, pasta_met):
    met = pd.read_csv(glob.glob(os.path.join(pasta_met, "mode_*_obj.txt"))[0], sep=r"\s+").set_index("OBJECT_ID")
    with xr.open_dataset(glob.glob(os.path.join(pasta_met, "mode_*_obj.nc"))[0]) as nc:
        idf, ido = nc.fcst_obj_id.values, nc.obs_obj_id.values
        cf, co = nc.fcst_clus_id.values, nc.obs_clus_id.values
    mapf = {j: int(idf[a["mascara"]].max()) for j, a in enumerate(r.prev)}
    mapo = {i: int(ido[a["mascara"]].max()) for i, a in enumerate(r.obs)}
    n_f = int(met.index.str.match(r"^F\d+$").sum()); n_o = int(met.index.str.match(r"^O\d+$").sum())
    areas = all(met.loc[f"F{mapf[j]:03d}", "AREA"] == a["area"] for j, a in enumerate(r.prev)) and \
            all(met.loc[f"O{mapo[i]:03d}", "AREA"] == a["area"] for i, a in enumerate(r.obs))
    p = met[met.index.str.match(r"^F\d+_O\d+$")].INTEREST
    p = p[p >= 0]
    dif = [abs(p.get(f"F{mapf[j]:03d}_O{mapo[i]:03d}", np.nan) - r.interesse[i, j]) for i, j in r.detalhes]
    inv_f, inv_o = {v: k for k, v in mapf.items()}, {v: k for k, v in mapo.items()}
    mf, mo = np.zeros(len(r.prev)), np.zeros(len(r.obs))
    for oid, it in p.items():
        f, o = (int(x[1:]) for x in oid.split("_"))
        mf[inv_f[f]], mo[inv_o[o]] = max(mf[inv_f[f]], it), max(mo[inv_o[o]], it)
    mmi_met = float(np.median(np.concatenate([mf, mo]))) if len(mf) and len(mo) else 0.0
    cl = {}
    for j, a in enumerate(r.prev):
        c = int(cf[a["mascara"]].max())
        if c > 0: cl.setdefault(c, (set(), set()))[1].add(j)
    for i, a in enumerate(r.obs):
        c = int(co[a["mascara"]].max())
        if c > 0: cl.setdefault(c, (set(), set()))[0].add(i)
    cl_met = sorted((sorted(o), sorted(f)) for o, f in cl.values())
    return {"objetos_prev (nosso/MET)": f"{len(r.prev)}/{n_f}", "objetos_obs (nosso/MET)": f"{len(r.obs)}/{n_o}",
            "areas_iguais": areas, "pares_avaliados (nosso/MET)": f"{len(r.detalhes)}/{len(p)}",
            "dif_max_interesse": float(np.nanmax(dif)) if dif else 0.0,
            "MMI (nosso/MET)": f"{r.mmi:.4f}/{mmi_met:.4f}",
            "clusters_iguais": cl_met == sorted((c["obs"], c["prev"]) for c in r.clusters)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs"); ap.add_argument("var_obs"); ap.add_argument("prev"); ap.add_argument("var_prev")
    ap.add_argument("--met", required=True, help="executável mode do MET")
    ap.add_argument("--met-config", required=True, help="MODEConfig_default da instalação do MET")
    ap.add_argument("--config", choices=["padrao", "dissertacao"], default="padrao")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()

    cfg = ModeConfig() if a.config == "padrao" else ModeConfig.dissertacao()
    pasta = a.saida or tempfile.mkdtemp(prefix="compara_met_")
    os.makedirs(pasta, exist_ok=True)
    obs, prev = le(a.obs, a.var_obs), le(a.prev, a.var_prev)
    escreve_cf(obs, os.path.join(pasta, "obs.nc")); escreve_cf(prev, os.path.join(pasta, "prev.nc"))
    open(os.path.join(pasta, "MODEConfig"), "w").write(config_met(cfg, a.met_config))
    subprocess.run([a.met, os.path.join(pasta, "prev.nc"), os.path.join(pasta, "obs.nc"),
                    os.path.join(pasta, "MODEConfig"), "-outdir", pasta, "-v", "1"], check=True,
                   env=dict(os.environ, OMP_NUM_THREADS="1"), stdout=subprocess.DEVNULL)
    r = avaliar(obs.values, prev.values, cfg)          # NaN = dado ausente, tratado como no MET
    for k, v in compara(r, pasta).items():
        print(f"{k:30s} {v}")
    print(f"(saída do MET em {pasta})")
