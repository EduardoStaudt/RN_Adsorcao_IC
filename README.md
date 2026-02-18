# Rede Neural para Predição Rápida de Desempenho em Colunas de Adsorção em Leito Fixo

Este repositório reúne o código do projeto de Iniciação Científica que desenvolve um **modelo de rede neural (RNA)** para **predizer rapidamente o desempenho de colunas de adsorção em leito fixo**.

A ideia central é substituir (ou complementar) simulações numéricas demoradas do **modelo matemático de advecção–dispersão com cinética de adsorção** por uma **rede neural treinada** em dados gerados por simulação e/ou experimento. Isso reduz drasticamente o tempo de cálculo e viabiliza inferência em tempo quase real (ex.: análise rápida de cenários, estimativas de breakthrough e perfis ao longo da coluna).

---

## 🎯 Objetivos

- Construir uma base de dados com condições operacionais, propriedades do sistema e respostas da coluna (ex.: **curvas de breakthrough** e perfis no leito).
- Treinar uma **rede neural artificial (RNA)** para aproximar o comportamento do modelo matemático de referência.
- Avaliar o desempenho do modelo em termos de **erro de predição** e **tempo de inferência**, comparando com o modelo convencional.
- Manter uma base de código organizada e reprodutível para reutilização/expansão em trabalhos futuros.

---

## 📌 Modelos neste repositório

### 1) `adsorption_nn` (modelo principal) — **ADSORÇÃO (22 → 208)**

O modelo recebe **22 parâmetros de entrada** e retorna **208 saídas**, sendo:

- **4 saídas finais**: `C_out_final`, `q_out_final`, `T_out_final`, `N_ads_final`
- **204 saídas em perfis (4 blocos × 51 pontos)**:
  - `C_z(51)`   : concentração ao longo do leito  
  - `q_z(51)`   : carga adsorvida ao longo do leito  
  - `T_z(51)`   : temperatura ao longo do leito  
  - `Qtot_t(51)`: grandeza temporal agregada (conforme o dataset)

### 2) `nu_uvec_nn` — **NU_UVEC**

Modelo auxiliar do projeto (mantido separado), com validação em dois métodos para comparação.

---

## 📂 Estrutura do Repositório (organização atual)

```text
├── data/
│   ├── raw/
│   │   ├── adsorption/            # (não versionado) batches/arquivos grandes
│   │   └── nu_uvec/
│   └── processed/
│       ├── adsorption/
│       │   ├── dataset_FULL.npz   # dataset final consolidado (usado no treino/validação)
│       │   └── dataset_FULL.csv   # opcional (pode ser removido se ficar grande)
│       └── nu_uvec/
│           └── dataset_nu_Uvec_analitico.csv
│
├── models/                        # (não versionado) modelos treinados e scalers
│   ├── adsorption/
│   └── nu_uvec/
│
├── outputs/                       # (não versionado) resultados e logs de validação/plots
│   ├── adsorption/
│   │   ├── training/
│   │   └── inference/
│   └── nu_uvec/
│       ├── training/
│       └── inference/
│
├── notebooks/
│   └── Visualization.ipynb
│
├── scripts/
│   └── download_dataset_adsorption.ps1   # (opcional) helper para baixar o dataset grande
│
├── src/
│   ├── adsorption_nn/
│   │   ├── config.py
│   │   ├── dataset_build.py
│   │   ├── train.py
│   │   ├── validate_masked_mape.py
│   │   ├── validate_eps_mape.py
│   │   ├── compare_validations.py
│   │   └── gui_flet.py
│   └── nu_uvec_nn/
│       ├── config.py
│       ├── train.py
│       ├── train_legacy.py
│       ├── validate_masked_mape.py
│       ├── validate_eps_mape.py
│       └── compare_validations.py
│
├── run.ps1
├── requirements.txt
└── README.md
````

> **Nota:** As pastas **models/** e **outputs/** (e artefatos de treino/validação) ficam fora do versionamento para evitar poluir o repositório com arquivos grandes. O código e os scripts de reprodução ficam disponíveis.

---

## 🧪 Principais Tecnologias

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn / Joblib (normalização e persistência de scalers)
* (Opcional) Keras-Tuner (busca de hiperparâmetros)
* (Opcional) Flet (interface simples)

---

## ▶️ Como Rodar (clone → instalar → dataset → treinar → validar)

### 1) Instalar dependências

Dentro da pasta do projeto:

```bash
pip install -r requirements.txt
```

> Recomendado: usar ambiente virtual (`venv`).

---

## 📦 Dataset grande (ADSORÇÃO)

O `dataset_FULL.npz` **não é versionado** (é grande e excede limites do GitHub).
Você deve baixá-lo e colocar em:

```
data/processed/adsorption/dataset_FULL.npz
```

### Opção A) Download manual

* Baixe pelo link (Drive/host) informado no Release/README do projeto.
* Coloque o arquivo em `data/processed/adsorption/`.

### Opção B) Script PowerShell (Windows)

Se você estiver usando um script em `scripts/download_dataset_adsorption.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_dataset_adsorption.ps1
```

---

# ✅ ADSORÇÃO (22 → 208)

## 2) Treinar

```bash
python src/adsorption_nn/train.py
```

Isso gera (localmente):

* `models/adsorption/best_model.keras`
* `models/adsorption/scaler_input.save`
* `models/adsorption/scaler_output.save`
* `models/adsorption/model_meta.json`

## 3) Validar com dois métodos (EPS e MASKED) e comparar

No **PowerShell (VS Code / Windows)**:

```powershell
$tag = (Get-Date -Format "yyyyMMdd_HHmmss")

python src/adsorption_nn/validate_masked_mape.py --tag $tag --seed 42
python src/adsorption_nn/validate_eps_mape.py    --tag $tag --seed 42

python src/adsorption_nn/compare_validations.py  --tag $tag
```

Saídas típicas:

* `outputs/adsorption/inference/<TAG>/masked/`
* `outputs/adsorption/inference/<TAG>/eps/`
* `outputs/adsorption/inference/<TAG>/compare_validations_<TAG>.csv`

Além disso, existe `outputs/adsorption/inference/LATEST.txt`, apontando a última execução.

---

# ✅ NU_UVEC

## Treinar

```bash
python src/nu_uvec_nn/train.py
```

## Validar e comparar (EPS vs MASKED)

```powershell
$tag = (Get-Date -Format "yyyyMMdd_HHmmss")

python src/nu_uvec_nn/validate_masked_mape.py --tag $tag --seed 42
python src/nu_uvec_nn/validate_eps_mape.py    --tag $tag --seed 42

python src/nu_uvec_nn/compare_validations.py  --tag $tag
```

---

## 📌 Sobre os dois métodos de validação (EPS vs MASKED)

Este projeto mantém **dois métodos** porque algumas variáveis podem ter muitos valores próximos de zero.

### 1) MAPE com EPS (estilo `vali.py`)

* Usa um `eps` no denominador para evitar divisão por zero.
* Pode gerar **MAPE gigantesco** quando `y_true` é muito pequeno (efeito de escala).

### 2) MAPE Mascarado (masked)

* **Ignora** pontos onde `|y_true| < threshold`.
* Geralmente gera um MAPE **mais interpretável** quando existem muitos zeros/próximos de zero.

O script `compare_validations.py` compara os resultados dos dois métodos para o mesmo conjunto.

---

## 🧾 Referências

* SHAFEYAN, M. et al. (2014). *(artigo base usado como referência metodológica no projeto)*

---

## 👨‍💻 Autor

* Eduardo Andrei Staudt – aluno de IC – Universidade Tecnológica Federal do Paraná (UTFPR)
* Contato: [edusta@alunos.utfpr.edu.br](mailto:edusta@alunos.utfpr.edu.br)
