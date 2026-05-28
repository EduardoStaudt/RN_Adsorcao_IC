# Rede Neural para Predição Rápida de Desempenho em Colunas de Adsorção em Leito Fixo

Este repositório contém o código do projeto de Iniciação Científica que desenvolve uma **rede neural (RNA)** para **predizer rapidamente** o desempenho de **colunas de adsorção em leito fixo**.

A proposta é aproximar o comportamento do **modelo matemático** (advecção–dispersão + cinética de adsorção), reduzindo o tempo de cálculo e permitindo inferência rápida (curvas de breakthrough, perfis ao longo do leito e variáveis finais).

---

## O que existe aqui

### `adsorption_nn` (principal)
- **Entradas:** 22 parâmetros  
- **Saídas:** 208 valores  
  - **4 finais:** `C_out_final`, `q_out_final`, `T_out_final`, `N_ads_final`  
  - **Perfis (4×51):** `C_z(51)`, `q_z(51)`, `T_z(51)`, `Qtot_t(51)`

### `nu_uvec_nn` (auxiliar)
Modelo menor para experimentos/validação (dataset analítico).

---

## Estrutura

```text
data/
  processed/
    adsorption/   (dataset FULL via download)
    nu_uvec/      (CSV analítico versionado)
models/           (gerado - não versionado)
outputs/          (gerado - não versionado)
src/
  adsorption_nn/
  nu_uvec_nn/
scripts/
requirements.txt
run.ps1
README.md
````

---

## Dataset (ADSORÇÃO) — download via Google Drive

O `dataset_FULL.npz` não é versionado no GitHub (arquivo grande).
O arquivo deve ficar em:

`data/processed/adsorption/dataset_FULL.npz`

### Download automático (recomendado)

1. Instale as dependências (inclui `gdown`):

```bash
pip install -r requirements.txt
```

2. Baixe o dataset:

* Se o script já estiver com o `DEFAULT_FILE_ID` configurado, você pode rodar **sem parâmetros**:

```bash
python scripts/download_adsorption_dataset.py
```

* Ou, passando um `file_id` manualmente:

```bash
python scripts/download_adsorption_dataset.py --file_id <FILE_ID_DO_DRIVE>
```

---

## Como rodar (Windows / macOS / Linux)

> Recomendado: Python 3.9+.

### 1) Criar ambiente virtual e instalar dependências

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

**macOS/Linux (bash/zsh)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Baixar dataset FULL (ADSORÇÃO)

```bash
python scripts/download_adsorption_dataset.py
```

---

## ADSORÇÃO (22 → 208)

### Treinar

```bash
python src/adsorption_nn/train.py
```

### Validar (dois métodos) + comparar

**Windows (PowerShell)**

```powershell
$tag = (Get-Date -Format "yyyyMMdd_HHmmss")
python src/adsorption_nn/validate_masked_mape.py --tag $tag --seed 42
python src/adsorption_nn/validate_eps_mape.py    --tag $tag --seed 42
python src/adsorption_nn/compare_validations.py  --tag $tag
```

**macOS/Linux**

```bash
tag=$(date +"%Y%m%d_%H%M%S")
python src/adsorption_nn/validate_masked_mape.py --tag "$tag" --seed 42
python src/adsorption_nn/validate_eps_mape.py    --tag "$tag" --seed 42
python src/adsorption_nn/compare_validations.py  --tag "$tag"
```

Resultados:

* `outputs/adsorption/inference/<TAG>/masked/`
* `outputs/adsorption/inference/<TAG>/eps/`
* `outputs/adsorption/inference/<TAG>/compare_validations_<TAG>.csv`

---

## NU_UVEC

### Treinar

```bash
python src/nu_uvec_nn/train.py
```

### Validar + comparar

**Windows (PowerShell)**

```powershell
$tag = (Get-Date -Format "yyyyMMdd_HHmmss")
python src/nu_uvec_nn/validate_masked_mape.py --tag $tag --seed 42
python src/nu_uvec_nn/validate_eps_mape.py    --tag $tag --seed 42
python src/nu_uvec_nn/compare_validations.py  --tag $tag
```

**macOS/Linux**

```bash
tag=$(date +"%Y%m%d_%H%M%S")
python src/nu_uvec_nn/validate_masked_mape.py --tag "$tag" --seed 42
python src/nu_uvec_nn/validate_eps_mape.py    --tag "$tag" --seed 42
python src/nu_uvec_nn/compare_validations.py  --tag "$tag"
```

---

## Sobre EPS vs MASKED (MAPE)

Alguns alvos têm muitos valores próximos de zero:

* **MAPE com EPS** (denominador com `eps`): evita divisão por zero, mas pode explodir quando `y_true` é muito pequeno.
* **MAPE mascarado**: ignora pontos com `|y_true| < threshold`, mantendo o MAPE mais interpretável quando há muitos zeros/quase-zeros.

A comparação (`compare_validations.py`) coloca os dois lado a lado.

---

## Autor

* Eduardo Andrei Staudt — UTFPR
* Contato: [edusta@alunos.utfpr.edu.br](mailto:edusta@alunos.utfpr.edu.br)
## Orientador: 
* Evandro Alves Nakajima — UTFPR