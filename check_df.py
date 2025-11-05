
# CODIGGO DE CHECAGEM DE SEVIO PADRAO DO DATASET GERADO

import pandas as pd, numpy as np

df = pd.read_csv("dataset_nu_Uvec_analitico.csv")
y = df[[f"u_{k}" for k in range(25)]].to_numpy()
print("Desvio-padrão médio das colunas:", y.std(axis=0).mean())