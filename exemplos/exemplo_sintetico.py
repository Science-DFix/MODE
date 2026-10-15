"""Exemplo sem dados externos: dois sistemas de chuva sintéticos, a previsão deslocada e com intensidade diferente.

Uso:  python exemplos/exemplo_sintetico.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mode_python import ModeConfig, avaliar, gss_objetos


def elipse(ny, nx, cy, cx, a, b, ang_graus, pico):
    """Chuva (mm) com forma elíptica e máximo `pico` no centro."""
    y, x = np.mgrid[0:ny, 0:nx]
    t = np.radians(ang_graus)
    u = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    v = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return pico * np.exp(-((u / a) ** 2 + (v / b) ** 2))


ny, nx = 126, 102                                   # grade de 0,6° da América do Sul (60°S–15°N, 90°W–30°W)
obs = elipse(ny, nx, 60, 70, 18, 5, -40, 40) + elipse(ny, nx, 95, 30, 6, 6, 0, 25)
prev = elipse(ny, nx, 64, 66, 20, 6, -35, 30) + elipse(ny, nx, 30, 80, 5, 4, 0, 20)

r = avaliar(obs, prev, ModeConfig())                # configuração padrão do MET
print(f"Objetos: {len(r.obs)} observados, {len(r.prev)} previstos")
for i, j, it in r.pares:
    d = r.detalhes[i, j]
    print(f"  par O{i + 1}–F{j + 1}: interesse {it:.3f} | centroides a {d['dist_centroide_km']:.0f} km | "
          f"ângulo {d['dif_angulo']:.1f}° | razão de áreas {d['razao_area']:.2f}")
print(f"MMI = {r.mmi:.3f} (prev {r.mmif:.3f}, obs {r.mmio:.3f}) | GSS de objetos = {gss_objetos(r):.3f}")
print(f"Clusters: {r.clusters}")
