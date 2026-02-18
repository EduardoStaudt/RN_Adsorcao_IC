# Rede Neural para Predição Rápida de Desempenho em Colunas de Adsorção em Leito Fixo

Este repositório contém o código e os dados do projeto de Iniciação Científica que desenvolve um **modelo de rede neural (RNA)** para **prever rapidamente o desempenho de colunas de adsorção em leito fixo**.

A ideia central é substituir (ou complementar) simulações numéricas demoradas do **modelo matemático de advecção–dispersão com cinética de adsorção** por uma **rede neural treinada** em dados gerados por simulação e/ou experimento. Com isso, é possível reduzir drasticamente o tempo de cálculo e viabilizar aplicações de inferência em tempo quase real (por exemplo, para análise rápida de cenários, estimativas de curvas de breakthrough e perfis ao longo da coluna).

---

## 🎯 Objetivos

- Construir uma base de dados com condições operacionais, propriedades do sistema e respostas da coluna (por exemplo, **curvas de breakthrough** e perfis ao longo do leito).
- Treinar uma **rede neural artificial (RNA)** para aproximar o comportamento do modelo matemático de referência.
- Avaliar o desempenho do modelo em termos de **erro de predição** e **tempo de inferência**, comparando com o modelo convencional.
- Manter uma base de código organizada e reprodutível para reutilização/expansão em trabalhos futuros.

---

## 📌 O que a rede principal prevê (ADSORÇÃO)

O modelo `adsorption_nn` foi construído para receber **22 parâmetros de entrada** e retornar **208 saídas**, sendo:

- **4 saídas finais**: `C_out_final`, `q_out_final`, `T_out_final`, `N_ads_final`
- **204 saídas em perfis (4 blocos × 51 pontos)**:
  - `C_z(51)`  : concentração ao longo do leito
  - `q_z(51)`  : carga adsorvida ao longo do leito
  - `T_z(51)`  : temperatura ao longo do leito
  - `Qtot_t(51)`: grandeza temporal agregada (conforme dataset)

---

## 📂 Estrutura do Repositório (organização atual)

```text
├── data/
│   ├── raw/
│   │   ├── adsorption/         # (não versionado) batches/arquivos grandes
│   │   └── nu_uvec/
│   └── processed/
│       ├── adsorption/
│       │   ├── dataset_FULL.npz # dataset final consolidado (usado no treino/validação)
│       │   └── dataset_FULL.csv # opcional (pode ser removido se ficar grande)
│       └── nu_uvec/
│           └── dataset_nu_Uvec_analitico.csv
│
├── models/                      # (não versionado) modelos treinados e scalers
│   ├── adsorption/
│   └── nu_uvec/
│
├── outputs/                     # (não versionado) resultados e gráficos
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

> **Observação**: As pastas **models/** e **outputs/** e os arquivos gerados por treino e validação não são versionados (estão no .gitignore)**, para evitar poluição do repositório com arquivos grandes. O código-fonte e os scripts de geração estão disponíveis para reprodução.

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

## ▶️ Como Rodar (clone → instalar → treinar → validar)

### 1) Instalar dependências

No terminal, dentro da pasta do projeto:

```bash
pip install -r requirements.txt
```

---

## ✅ Rodando o modelo principal (ADSORÇÃO 22 → 208)

### 2) Treinar

```bash
python src/adsorption_nn/train.py
```

Isso vai gerar:

* `models/adsorption/best_model.keras`
* `models/adsorption/scaler_input.save`
* `models/adsorption/scaler_output.save`
* `models/adsorption/model_meta.json`

### 3) Validar com dois métodos (EPS e MASKED) e comparar

No PowerShell (VS Code), rode:

```powershell
$tag = (Get-Date -Format "yyyyMMdd_HHmmss")

python src/adsorption_nn/validate_masked_mape.py --tag $tag --seed 42
python src/adsorption_nn/validate_eps_mape.py    --tag $tag --seed 42

python src/adsorption_nn/compare_validations.py  --tag $tag
```

Os resultados ficam em:

* `outputs/adsorption/inference/<TAG>/masked/`
* `outputs/adsorption/inference/<TAG>/eps/`
* `outputs/adsorption/inference/<TAG>/compare_validations_<TAG>.csv`

---

## ✅ Rodando o modelo NU_UVEC

### Treinar

```bash
python src/nu_uvec_nn/train.py
```

### Validar e comparar (EPS vs MASKED)

```powershell
$tag = (Get-Date -Format "yyyyMMdd_HHmmss")

python src/nu_uvec_nn/validate_masked_mape.py --tag $tag --seed 42
python src/nu_uvec_nn/validate_eps_mape.py    --tag $tag --seed 42

python src/nu_uvec_nn/compare_validations.py  --tag $tag
```

---

## 📌 Sobre os dois métodos de validação (EPS vs MASKED)

Este projeto mantém dois métodos porque alguns alvos possuem muitos valores próximos de zero.

* **MAPE com EPS (estilo `vali.py`)**

  * Usa um `eps` no denominador para evitar divisão por zero.
  * Pode produzir MAPE gigantesco se `y_true` for muito pequeno (efeito de escala).

* **MAPE Mascarado (masked)**

  * Ignora pontos onde `|y_true| < threshold`.
  * Em geral, gera um MAPE mais interpretável quando existem muitos zeros no sinal.

A comparação automática (`compare_validations.py`) mostra, para o mesmo conjunto, como o MAPE muda entre os dois métodos.

---

## 🧾 Referências

* SHAFEYAN, M. et al. (2014). *(artigo base usado como referência metodológica no projeto)*

---

## 👨‍💻 Autor

* Eduardo Andrei Staudt – aluno de IC – Universidade Tecnológica Federal do Paraná (UTFPR)
* Contato: [edusta@alunos.utfpr.edu.br](mailto:edusta@alunos.utfpr.edu.br)

---
