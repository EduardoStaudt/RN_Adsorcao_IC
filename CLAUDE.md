# Rede Neural para predição de desempenho de colunas de adsorção em leito fixo

- Utilize o uv.lock para ver as versões dos pacotes e bibliotecas que estão sendo utilizados no projeto.
- Idioma de trabalho: **português brasileiro**.
- Nunca alterar a estrutura da rede (`build_model`) sem autorização explícita.
- Preservar todos os comentários e TODOs existentes no código.
- Mudar o mínimo possível do `train.py` a cada intervenção.
- Perguntar antes de qualquer alteração em arquivos principais.

---

## Descrição do projeto

O projeto tem como objetivo desenvolver uma solução computacional para prever, de forma rápida, o desempenho de colunas de adsorção em leito fixo. Essas colunas são usadas em processos de separação de gases, como a captura de CO₂, mas sua simulação tradicional pode ser complexa e demorada, pois envolve modelos matemáticos com vários fenômenos físicos, como transferência de massa, temperatura, equilíbrio de adsorção e queda de pressão.

Para resolver esse problema, será desenvolvida uma rede neural em Python/TensorFlow capaz de aprender o comportamento dessas colunas a partir de dados simulados. A ideia é que, depois de treinada, a rede consiga fazer previsões muito mais rápidas do que um modelo matemático convencional.

O projeto é composto por três entregas:
1. **Rede neural** (surrogate model) — `src/adsorption_nn/`
2. **Biblioteca de apoio** — `inference/predictor.py` e módulos em `src/`
3. **Interface visual** — `src/adsorption_nn/gui_flet.py` (Flet/Flutter)

---

## Estrutura de arquivos relevantes

```
src/adsorption_nn/
    train.py          ← script principal de treino (NÃO alterar sem confirmação)
    config.py         ← centraliza todos os caminhos do projeto
    gui_flet.py       ← interface gráfica (em progresso)
src/nu_uvec_nn/
    train.py          ← modelo auxiliar nu → u_vec (independente)
inference/
    predictor.py      ← API de inferência (AdsorptionPredictor)
scripts/
    clean_invalid_seeds.py  ← limpeza de seeds com N_ads_final ≈ 0
data/
    processed/adsorption/dataset_FULL.csv     ← 993.507 linhas após limpeza  →  cfg.ADS_FULL_CSV
    newdataset/SubSets/dataset_optuna_1000.csv   ← 995 linhas    →  cfg.ADS_SUB_1000
    newdataset/SubSets/dataset_optuna_10000.csv  ← 9.943 linhas  →  cfg.ADS_SUB_10000
    newdataset/SubSets/dataset_optuna_50000.csv  ← 49.721 linhas →  cfg.ADS_SUB_50000
models/adsorption/
    best_model.keras, scaler_input.save, scaler_output.save
    model_meta.json, best_hp.json
outputs/adsorption/training/optuna/
    optuna_adsorption.db   ← banco SQLite do Optuna (gerado ao rodar com USE_OPTUNA=True)
    opt_history.html, param_importance.html, parallel_coord.html
```

---

## Arquitetura atual da rede (build_model)

- **Entradas (X):** 22 parâmetros físicos do processo
- **Saídas (Y):** 157 valores (4 escalares finais + 51×3 perfis espaciais C_z, q_z, T_z)
- **Estrutura dinâmica:** `n_layers` camadas Dense com `n_units` neurônios cada, seguidas de Dropout, mais camada de saída linear
- **Assinatura atual:** `build_model(n_layers=3, n_units=352, activation="elu", dropout=0.10, l2_reg=1e-5, lr=5e-4)`
- **Loss:** MSE | **Métricas:** MAE, RMSE | **Otimizador:** Adam
- **Normalização:** Z-score (StandardScaler) para X e Y separadamente

---

## Otimização de hiperparâmetros (Optuna)

Substituiu o keras_tuner na seção 6 do `train.py`. Configuração atual:

- **Flag:** `USE_OPTUNA = True` (linha ~166) — mudar para `False` para treinar com arquitetura fixa
- **Trials:** 64 (2^6 — um por HP) | **Epochs por trial:** 200 | **Pruner:** MedianPruner (n_startup=5, warmup=10)
- **Storage:** SQLite em `outputs/adsorption/training/optuna/optuna_adsorption.db`
- **Dashboard em tempo real:** `optuna-dashboard sqlite:///outputs/adsorption/training/optuna/optuna_adsorption.db` → `http://localhost:8080`
- **Gráficos HTML** gerados automaticamente após o optimize (opt_history, param_importance, parallel_coord)

**Espaço de busca atual:**
| HP | Tipo | Range |
|---|---|---|
| n_layers | int | 2 – 4 |
| n_units | int | 64 – 256 (step 32) |
| activation | categorical | relu, elu |
| dropout | float | 0.0 – 0.2 (step 0.05) |
| l2_reg | float log | 1e-6 – 1e-3 |
| lr | float log | 1e-4 – 1e-2 |

---

## Datasets — estado após limpeza (2026-05-17)

Critério de remoção: `np.isclose(N_ads_final, 0, atol=1e-9)` — seeds com adsorção degenerada.

| Arquivo | Antes | Removidas | Após |
|---|---|---|---|
| dataset_FULL.csv | 999.001 | 5.494 | **993.507** |
| dataset_optuna_1000.csv | 1.000 | 5 | **995** |
| dataset_optuna_10000.csv | 10.000 | 57 | **9.943** |
| dataset_optuna_50000.csv | 50.000 | 279 | **49.721** |

Script reutilizável: `python scripts/clean_invalid_seeds.py [--dry-run]`

**IMPORTANTE — ao treinar com os subsets, sempre usar os caminhos do `config.py`:**
```python
import adsorption_nn.config as cfg
# cfg.ADS_SUB_1000   → dataset_optuna_1000.csv   (995 linhas)
# cfg.ADS_SUB_10000  → dataset_optuna_10000.csv  (9.943 linhas)
# cfg.ADS_SUB_50000  → dataset_optuna_50000.csv  (49.721 linhas)
# cfg.ADS_FULL_CSV   → dataset_FULL.csv          (993.507 linhas)
```
Nunca hardcodar caminhos de dataset — centralizado em `src/adsorption_nn/config.py`.

---

## Dependências principais (pyproject.toml)

```
tensorflow>=2.20,<3
optuna>=3.6
optuna-dashboard>=0.9
plotly>=5.0
keras-tuner>=1.4   (mantido no lock, não mais usado ativamente)
scikit-learn>=1.4
pandas>=2.2
numpy>=1.26
joblib>=1.3
matplotlib>=3.8
flet>=0.24
flask>=3.0
```

---

## Histórico de tarefas

### ✅ CONCLUÍDAS

**TAREFA 1 — Remoção de seeds inválidas do dataset**
- Criado `scripts/clean_invalid_seeds.py` com `np.isclose(atol=1e-9)` (não `== 0`)
- Removidas 5.494 linhas do FULL e proporcionalmente nos subsets
- Script aceita `--dry-run` para inspeção sem alteração

**TAREFA 2 — Substituição do keras_tuner pelo Optuna**
- Somente a seção 6 do `train.py` foi alterada (seções renumeradas: 6→Optuna, 7→Treino, 8→Curva)
- `build_model()` refatorada para construção dinâmica com `n_layers` e `n_units`
- Novo espaço de busca: n_layers, n_units, activation, dropout, l2_reg (log), lr (log)
- Pruning com `MedianPruner`
- Storage SQLite para dashboard em tempo real
- Gráficos HTML gerados automaticamente pós-optimize
- `optuna`, `optuna-dashboard`, `plotly` adicionados ao `pyproject.toml`

---

### 🔲 PENDENTES

**TAREFA 3 — Testar Optuna nos mini datasets (`tune_subsets.py`)**
- Criar `scripts/tune_subsets.py` separado do `train.py`
- Para cada subset em `data/newdataset/SubSets/`: rodar Optuna, salvar melhores HPs em JSON identificado pelo nome do subset
- Ao final, gerar tabela comparando os melhores HPs entre os subsets
- Objetivo: verificar se os HPs variam com o tamanho do dataset

**TAREFA 4 — Laboratório de experimentos (log CSV/JSON)**
- Para cada execução registrar: nome do subset, tamanho, melhores HPs, melhor val_rmse, tempo de busca
- Arquivo de log acumulativo (CSV ou JSON Lines)
- Servirá de base para o artigo científico

---

## Base bibliográfica

- Shafeeyan et al. 2014 — modelagem matemática de colunas de leito fixo para adsorção de CO₂ (balanços de material, cinética LDF, energia, momento, isotermas, dispersão, Ergun, Danckwerts)
- Wakao & Funazkri — dispersão axial
- Materiais de estudo internos: MLP, backpropagation, funções de ativação, TensorFlow 2.0

---

## Contexto transferido do ChatGPT (preservado)

**Identity**
- Idioma de trabalho: português brasileiro
- Interesse técnico/acadêmico: redes neurais, modelagem matemática, adsorção em leito fixo, CO₂, PDEs/EDPs, LDF, Danckwerts, Ergun, TensorFlow e aprendizado de máquina aplicado à engenharia química

**Projects (decisões técnicas registradas)**
- Geração de dataset massivo por simulações 1D de massa/energia/momento
- Condições de contorno de Danckwerts, cinética LDF, queda de pressão por Ergun, dispersão axial por Wakao/Funazkri
- Normalização por números adimensionais
- Comparação de MLP e CNN-1D, penalizações de consistência física
- Validação fora do regime e incerteza por ensembles/MC-dropout
