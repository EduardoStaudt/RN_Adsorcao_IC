# Contrato de Predição (RNA Adsorção) — FRONT ⇄ BACK

Este documento define o **formato JSON** trocado entre o **Flutter (front-end)** e o **backend Python** que carrega a **rede neural de adsorção**.

A ideia é que o Flutter envie os **22 parâmetros de entrada (2 → 23)** e receba:

- **4 valores finais** (`*_final`)
- **séries/perfis** como vetores (`*_points`)

> **Padrão de nomes:**
> - **Escalar:** `*_final` (um único valor)
> - **Vetor/lista:** `*_points` (vários valores)

---

## Endpoints

### `GET /health`
Retorna um JSON simples indicando que o serviço está ativo.

### `POST /predict`
Recebe um JSON com os **22 inputs** e retorna um JSON com os resultados.

- **Content-Type:** `application/json`
- **Body:** JSON (flat) com as chaves abaixo.

---

## Inputs (22 parâmetros) — chaves científicas

> Entradas da rede: **do 2 ao 23**.

| Chave | Descrição (PT-BR) | Unidade | Tipo |
|---|---|---:|---:|
| `L` | Comprimento do leito | m | float |
| `Nz` | Número de pontos no leito (malha em z) | – | int/float |
| `eps` | Porosidade do leito | – | float |
| `rho_b` | Densidade aparente do adsorvente | kg/m³ | float |
| `u` | Velocidade do fluido | m/s | float |
| `D_ax` | Coef. difusão/dispersão axial | m²/s | float |
| `kl` | Coef. transferência de massa | 1/s | float |
| `qmax` | Capacidade máxima de adsorção | mol/kg | float |
| `b` | Constante de afinidade | 1/concentração | float |
| `n` | Expoente de heterogeneidade (Freundlich) | – | float |
| `lam_z` | Condutividade / dispersão térmica axial | W/m·K | float |
| `rho_g` | Densidade do gás | kg/m³ | float |
| `cp_g` | Capacidade térmica do gás | J/kg·K | float |
| `cp_s` | Capacidade térmica do sólido | J/kg·K | float |
| `D_col` | Diâmetro interno da coluna | m | float |
| `h_w` | Coef. transferência de calor para a parede | W/m²·K | float |
| `T_wall` | Temperatura da parede | K | float |
| `dH` | Calor de adsorção (tipicamente negativo) | J/mol | float |
| `dt` | Intervalo de tempo | s | float |
| `t_end` | Tempo final da simulação | s | float |
| `C_in` | Concentração na entrada | mol/m³ (etc.) | float |
| `T_in` | Temperatura na entrada | K | float |

### Observação sobre compatibilidade do modelo atual
O modelo treinado existente usa internamente as chaves `rho_B` e `kL`.
O backend aceita `rho_b` e `kl` (padrão do front) e faz o mapeamento automaticamente.

---

## Outputs — formato do `POST /predict`

### Escalares (tempo final)
| Chave | Descrição | Tipo |
|---|---|---:|
| `C_out_final` | Concentração na saída no tempo final | float |
| `q_out_final` | Carga adsorvida (saída ou média) no tempo final | float |
| `T_out_final` | Temperatura na saída no tempo final | float |
| `N_ads_final` | Quantidade total adsorvida no leito no tempo final | float |

### Vetores (séries/perfis)
| Chave | Descrição | Tamanho |
|---|---|---:|
| `t_points` | eixo do tempo para séries | 51 |
| `C_out_points` | curva de breakthrough `C_out(t)` | 51 |
| `Qtot_points` | quantidade total adsorvida `Qtot(t)` | 51 |
| `z_points` | eixo do leito (posição z) | 51 |
| `C_z_points` | concentração no leito ao longo de z (no snapshot previsto) | 51 |
| `q_z_points` | adsorvido no sólido ao longo de z | 51 |
| `T_z_points` | temperatura ao longo de z | 51 |

> **Nota (MVP):** o modelo atual prevê diretamente `C_z`, `q_z`, `T_z` e `Qtot(t)`.
> A série `C_out_points` pode ser um **placeholder** (ex.: interpolação simples até `C_out_final`) até a RNA passar a prever explicitamente a curva `C_out(t)`.

---

## Exemplo — Request (Flutter → API)

```json
{
  "L": 1.20,
  "Nz": 51,
  "eps": 0.38,
  "rho_b": 650.0,
  "u": 0.20,
  "D_ax": 1.0e-4,
  "kl": 0.12,
  "qmax": 4.2,
  "b": 2.0e-5,
  "n": 0.85,
  "lam_z": 0.25,
  "rho_g": 1.18,
  "cp_g": 1005.0,
  "cp_s": 920.0,
  "D_col": 0.10,
  "h_w": 40.0,
  "T_wall": 298.15,
  "dH": -28000.0,
  "dt": 10.0,
  "t_end": 3600.0,
  "C_in": 1.50,
  "T_in": 298.15
}

## Exemplo — Response (API → Flutter)
{
  "C_out_final": 0.12,
  "q_out_final": 3.95,
  "T_out_final": 301.7,
  "N_ads_final": 12.4,

  "t_points": [0, 10, 20, 30],
  "C_out_points": [0.0, 0.01, 0.03, 0.08],
  "Qtot_points": [0.0, 1.2, 2.4, 3.1],

  "z_points": [0.0, 0.03, 0.06, 0.09],
  "C_z_points": [1.2, 1.0, 0.8, 0.2],
  "q_z_points": [4.1, 4.0, 3.8, 2.5],
  "T_z_points": [298.2, 299.0, 300.5, 301.7]
}


---

## 4) `run.sh` (Mac/Linux) — opcional mas recomendado
```bash
#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   ./run.sh api
#   ./run.sh gui
#   ./run.sh train
#   ./run.sh build-data

cmd="${1:-}"

case "$cmd" in
  build-data)
    python src/adsorption_nn/dataset_build.py
    ;;
  train)
    python src/adsorption_nn/train.py
    ;;
  validate)
    tag=$(date +"%Y%m%d_%H%M%S")
    python src/adsorption_nn/validate_masked_mape.py --tag "$tag" --seed 42
    python src/adsorption_nn/validate_eps_mape.py    --tag "$tag" --seed 42
    python src/adsorption_nn/compare_validations.py  --tag "$tag"
    ;;
  gui)
    python src/adsorption_nn/gui_flet.py
    ;;
  api)
    python api/app.py
    ;;
  *)
    echo "Uso: ./run.sh {build-data|train|validate|gui|api}"
    exit 1
    ;;
esac