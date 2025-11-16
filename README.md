# Rede Neural para Predição Rápida de Desempenho em Colunas de Adsorção em Leito Fixo

Este repositório contém o código e os dados do projeto de Iniciação Científica que desenvolve um **modelo de rede neural** para **prever rapidamente o desempenho de colunas de adsorção em leito fixo**.

A ideia central é substituir (ou complementar) simulações numéricas demoradas do **modelo matemático de advecção–dispersão com cinética de adsorção** por uma **rede neural treinada** em dados gerados por simulação e/ou experimento, reduzindo o tempo de cálculo e permitindo uso em aplicações em tempo quase real.

---

## 🎯 Objetivos

- Construir uma base de dados com condições operacionais, propriedades do sistema e respostas da coluna (por exemplo, **curvas de breakthrough**).
- Treinar uma **rede neural artificial (RNA)** para aproximar o modelo matemático de referência.
- Avaliar o desempenho do modelo em termos de **erro de predição** e **tempo de inferência**, comparando com o modelo convencional.
- Criar uma base de código organizada para ser reutilizada/extendida em trabalhos futuros.

---
## 📂 Estrutura do Repositório
```text
├── data/
│   ├── raw/           # Dados brutos (CSV etc.)
│   ├── processed/     # Dados já tratados/normalizados
├── models/
│   ├── saved/         # Modelos treinados (.h5, .keras, .pb...)
├── notebooks/         # Notebooks de exploração e testes
├── src/
├── README.md
├── requirements.txt   # Dependências do projeto

---
## 🧪 Principais Tecnologias
- Python 3.11.9
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib / Seaborn (visualização de curvas e métricas)

(Opcional) Scikit-learn para normalização, divisão treino/teste etc.

---
## 🧾 Referências
SHAFEYAN, M. et al. Título do artigo. Nome da Revista, ano, páginas.

---
## 👨‍💻 Autor
- Eduardo Andrei Staudt – aluno de IC – Universidade Tecnológica Federal do Paraná (UTFPR)
- Contato: edusta@alunos.utfpr.edu.br






