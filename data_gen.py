import numpy as np
import pandas as pd
import math

# ===================== CONFIGURAÇÕES =====================
nx = ny = 5         # tamanho da malha 5×5
t1 = 1.0            # instante de tempo (fixo)
nu_ini = 0.001      # ν inicial
nu_fim = 2.0        # ν final
passo = 0.001       # passo de variação de ν
arquivo_saida = "dataset_nu_Uvec_analitico.csv"

# ===================== FUNÇÃO ANALÍTICA =====================
# def T_exact(x, y, t, nu_val):
    # Função base ajustada para depender linearmente de ν=nu.
    # return nu_val * math.tanh(t) * (math.sin(math.pi * x) + math.cos(math.pi * y))
    # correção aqui antes nao tinha a multiplicação por nu_val

# Vetoriza a função para operar em arrays numpy
# T_exact_vec = np.vectorize(T_exact)

# ===================== MALHA ESPACIAL =====================
x = np.linspace(0, 1, nx) # linspace gera 5 pontos entre 0 e 1
y = np.linspace(0, 1, ny) # linspace gera 5 pontos entre 0 e 1
X, Y = np.meshgrid(x, y, indexing="ij")  # X,Y (5x5) representam as coordenadas da malha

# ===================== BASE INDEPENDENTE DE ν =====================
# Para cada ν, apenas multiplicamos esta base
T_base = math.tanh(t1) * (np.sin(math.pi * X) + np.cos(math.pi * Y))  # (5,5)

# ===================== LOOP SOBRE ν =====================
# np.arange gera valores de nu de nu_ini(0.001) a nu_fim(2.0) + 1e-12(para nao causar problemas de precisão por conta de arredondamento) com passo(0.001) definido.
nus = np.arange(nu_ini, nu_fim + 1e-12, passo, dtype=float) 
rows = []

for nu in nus:
    # Assim geraria tambem mas é mais lento
    # U = T_exact_vec(X, Y, t1, nu)  # calcula o campo 5x5 inteiro para esse ν
    U_vec = (nu * T_base).reshape(-1)  # 25 valores (flatten=vetorizar)
    rows.append([nu] + U_vec.tolist()) # tolist() converte array numpy para lista python e append adiciona na lista rows

# ===================== MONTAR DATAFRAME =====================
colunas = ["nu"] + [f"u_{k}" for k in range(25)] # gera as colunas com seus respectivos nomes das colunas
df = pd.DataFrame(rows, columns=colunas) # Cria o dataframe com as colunas geradas acima com o pandas.DataFrame

# ===================== SALVAR CSV =====================
# df.to_csv exporta o dataframe para um arquivo .csv o index é falso para não salvar o índice do pandas e o float_format define a precisão dos números salvos neste caso ate 10 casas decimais.
df.to_csv(arquivo_saida, index=False, float_format="%.10g")
print(f"\nDataset salvo em: {arquivo_saida}")
print(f"\nFormato: {df.shape}\n")
print(df.head())