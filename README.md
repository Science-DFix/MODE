# MODE em Python

Implementação enxuta, em Python puro (NumPy + SciPy), do **MODE** (*Method for Object-based Diagnostic
Evaluation*), o método de verificação por objetos de Davis et al. (2006, 2009). O mesmo método está no
**MET** (*Model Evaluation Tools*, DTC/NCAR).

O objetivo é ter o mesmo resultado do MET sem precisar instalar nem configurar o MET/METplus. Usa-se uma função
e uma configuração em Python. **A versão foi validada contra o executável `mode` do MET v12.2.1**: dá os mesmos objetos, pares,
interesses, MMI e clusters (ver [Validação](#validação)).

Foi desenvolvida para verificar previsões de chuva acumulada em 24 h sobre a América do Sul contra o MERGE/CPTEC,
numa dissertação de mestrado no INPE sobre assimilação de dados com o MPAS-JEDI. O código serve para qualquer
campo 2D numa grade regular.

---

## Como funciona

Para cada par de campos (observação × previsão):

1. **Objetos**
   - Média num disco de raio *R* (só com os pontos dentro da grade).
   - Limiar (≥) no campo suavizado.
   - Regiões conexas (8 vizinhos) e filtro opcional de área mínima.
2. **Atributos de cada objeto**
   - Área em pontos de grade.
   - Centroide, ângulo do eixo maior, comprimento e largura, calculados como no MET: sobre os vértices dos pixels.
   - Borda e percentis da chuva.
3. **Interesse de cada par**
   - Funções lineares por partes de cinco atributos: distância entre centroides, distância entre bordas, diferença de
     ângulo, razão de áreas e interseção ÷ menor área.
   - Combinação por média ponderada, com pesos e confianças:

     ```
     T = Σ wᵢ·cᵢ·Iᵢ / Σ wᵢ·cᵢ
     ```

   - A confiança no centroide é a razão de áreas. A confiança no ângulo vem da razão de aspecto dos dois objetos.
   - Pares com centroides além de `max_centroid_dist_km` não são avaliados.
4. **Pares e clusters**
   - Todo par com interesse ≥ `total_interest_thresh` é casado (equivale ao `MERGE_BOTH` do MET).
   - Os clusters juntam os pares e os objetos agrupados pelo limiar baixo (`merge_thresh`).
5. **Índices**
   - **MMI** (*median of maximum interest*): mediana dos interesses máximos de todos os objetos. Objeto sem par
     avaliado conta como 0.
   - MMIF e MMIO: o mesmo índice só para os objetos previstos ou só para os observados.
   - **GSS de objetos** (Davis et al. 2009, eqs. 2–4).

## Instalação

```bash
git clone https://github.com/Science-DFix/MODE.git
cd MODE
pip install -r requirements.txt      # Python ≥ 3.10; numpy e scipy (xarray, netCDF4 e pandas só para a validação)
```

O pacote é uma pasta, `mode_python/`. Basta tê-la no `PYTHONPATH`, ou rodar a partir da raiz do repositório.

## Uso

```python
import numpy as np
import xarray as xr
from mode_python import ModeConfig, avaliar, gss_objetos

# campos 2D na mesma grade; linha 0 = SUL (latitude crescente)
obs  = xr.open_dataset("MERGE_20260105.nc")["precip"].sortby("lat").values
prev = xr.open_dataset("previsao_20260105.nc")["precip"].sortby("lat").values

r = avaliar(obs, prev, ModeConfig())            # padrões do MET
print(r.mmi, r.mmif, r.mmio)                    # MMI, só previstos, só observados
print(len(r.obs), len(r.prev))                  # nº de objetos
for i, j, interesse in r.pares:                 # pares casados (índices a partir de 0)
    print(i, j, interesse, r.detalhes[i, j])    # componentes: distâncias, ângulo, razões
print(r.clusters)                               # [{"obs": [...], "prev": [...]}, ...]
print(gss_objetos(r))
```

Exemplo sem dados externos:

```bash
python exemplos/exemplo_sintetico.py
```

### O que `avaliar` devolve (`Resultado`)

| Campo | Conteúdo |
|---|---|
| `obs`, `prev` | lista de objetos; cada um é um dicionário com `area`, `centroide_x/y`, `angulo`, `comprimento`, `largura`, `razao_aspecto`, `p50`, `p90`, `soma`, `mascara` |
| `rotulos_obs`, `rotulos_prev` | matriz de rótulos (0 = fora de objeto, 1…n = objeto) |
| `interesse` | matriz `[n_obs, n_prev]`; `NaN` = par não avaliado |
| `detalhes[i, j]` | `interesse`, `dist_centroide_km`, `dist_borda_km`, `dif_angulo`, `razao_area`, `razao_intersecao` |
| `pares` | `(i_obs, j_prev, interesse)` com interesse ≥ limiar |
| `clusters` | grupos de objetos observados e previstos associados |
| `mmi`, `mmif`, `mmio` | índices MMI |

## Configuração

`ModeConfig()` traz os padrões do `MODEConfig_default` do MET v12.2.1. As distâncias estão em km (no MET, em pontos
de grade × `grid_res`).

| Parâmetro | Padrão (MET) | `ModeConfig.dissertacao()` | Significado |
|---|---|---|---|
| `grid_res_km` | 60 | 60 | tamanho do ponto de grade (km) |
| `conv_radius` | 1 | 3 | raio do disco de suavização (pontos); 0 = sem suavização |
| `conv_thresh` | 5,0 | 10,0 | limiar (≥) no campo suavizado |
| `vld_thresh` | 0,5 | 0,5 | fração mínima de dados válidos no disco (centro ausente) |
| `min_area` | 0 | 5 | área mínima do objeto (pontos) |
| `merge_thresh` | 1,25 | 3,0 | limiar baixo para agrupar objetos em clusters (`None` = sem merging) |
| `max_centroid_dist_km` | 800 | 1100 | pares mais distantes não são avaliados |
| `total_interest_thresh` | 0,70 | 0,55 | interesse mínimo para casar um par |
| `weight` | centroide 2 · borda 4 · ângulo 1 · área 1 · interseção 2 | 3 · 2,5 · 1 · 0,5 · 1,5 | pesos |
| `interest` | curvas do MET | curvas próprias | pontos (x, interesse) de cada função |

Qualquer item pode ser alterado:

```python
cfg = ModeConfig(conv_radius=2, conv_thresh=15.0, total_interest_thresh=0.6)
cfg.interest["centroid_dist"] = [(0, 1), (100, 1), (800, 0)]
```

**Convenções importantes**
- A matriz é `[linha, coluna] = [y, x]`, com a **linha 0 ao sul**, como o MET lê uma grade lat/lon regular. Se a
  latitude estiver decrescente, inverta (`sortby("lat")` ou `campo[::-1]`).
- A grade é tratada como **uniforme**: distâncias = pontos × `grid_res_km`, igual ao MET. Numa grade lat/lon, isso
  ignora o encolhimento da longitude com a latitude.
- `NaN` é dado ausente e entra na suavização como no MET.

## Validação

O pacote foi comparado com o executável `mode` do MET v12.2.1 (`validacao/compara_met.py`) em **60 casos**: chuva
24 h na grade de 0,6° da América do Sul (126 × 102), MERGE × 3 previsões × 10 dias, nas duas configurações acima.

| Configuração | Objetos (nº e áreas) iguais | Pares avaliados iguais | Maior diferença no interesse | Maior diferença no MMI | Clusters iguais |
|---|---|---|---|---|---|
| Padrão do MET | 30/30 | 30/30 | 5 × 10⁻⁶ | 5 × 10⁻⁶ | 30/30 |
| Dissertação | 30/30 | 30/30 | 5 × 10⁻⁶ | 4 × 10⁻⁶ | 30/30 |

As diferenças de 10⁻⁶ são o arredondamento do arquivo de texto do MET (5 decimais).

Para repetir com os seus dados (precisa do MET instalado):

```bash
python validacao/compara_met.py OBS.nc var_obs PREV.nc var_prev \
    --met /caminho/do/MET/bin/mode \
    --met-config /caminho/do/MET/share/met/config/MODEConfig_default \
    --config padrao
```

### Detalhes do MET que esta versão reproduz

Estes pontos não estão descritos no Guia do Usuário. Eles saem do código-fonte do MET e mudam o resultado:

- A suavização faz a média **só com os pontos dentro da grade**. Preencher com zero perto das bordas do domínio
  perde objetos.
- Centroide, ângulo e comprimento/largura usam os **vértices dos pixels** (um ponto conta se ele ou os vizinhos à
  esquerda, abaixo ou na diagonal inferior esquerda estão no objeto). A área é contada em pixels.
- A distância entre bordas é **0** quando os objetos se sobrepõem ou um está dentro do outro. Nos demais casos, é medida
  no contorno pelos vértices.
- A confiança no centroide é a **razão de áreas** (sem piso). A confiança no ângulo vale 1 para objetos em linha
  (razão de aspecto 0).
- A razão de aspecto é largura ÷ comprimento **medidos nos eixos principais**, e não a razão dos eixos da elipse de
  momentos.
- O merging por limiar não muda os objetos simples, só os clusters. Por isso, não muda o MMI.

## O que não está incluído

- MODE no tempo (MTD), MODE multivariado e o interesse de clusters.
- Os termos de interesse que têm peso 0 no padrão do MET: convex hull, diferença de razão de aspecto, curvatura,
  complexidade e razão de percentis.
- Figuras e arquivos de saída no formato do MET.

## Base

- Davis, C., B. Brown e R. Bullock, 2006: Object-based verification of precipitation forecasts. Part I: Methods
  and application to mesoscale rain areas. *Mon. Wea. Rev.*, **134**, 1772–1784, doi:10.1175/MWR3145.1.
- Davis, C. A., B. G. Brown, R. Bullock e J. Halley-Gotway, 2009: The Method for Object-Based Diagnostic
  Evaluation (MODE) applied to numerical forecasts from the 2005 NSSL/SPC Spring Program. *Wea. Forecasting*,
  **24**, 1252–1267, doi:10.1175/2009WAF2222241.1.
- DTC: *MET User's Guide*, versão 12.2 — capítulo "MODE Tool". https://met.readthedocs.io
- MET v12.2.1 (https://github.com/dtcenter/MET, licença Apache 2.0): o código-fonte foi usado **como referência**
  para conferir o algoritmo. Este repositório não contém código do MET.

## Autor

Diego Pereira — INPE, Programa de Pós-Graduação em Meteorologia.
