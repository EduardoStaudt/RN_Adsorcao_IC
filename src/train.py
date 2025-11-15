import numpy as np
from pathlib import Path 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import keras_tuner as kt
import joblib
import matplotlib.pyplot as plt

# ┌────────────── BLOCO DE PARÂMETROS (X da rede)───────────────┐
# │ L, Nz, eps, rho_B, u, D_ax, kL, qmax, b, n,                 │
# │ lam_z, rho_g, cp_g, cp_s, D_col, h_w, T_wall, dH,           │
# │ dt, t_end, C_in, T_in                                       │
# └─────────────────────────────────────────────────────────────┘

BASE_DIR = Path(__file__).resolve().parent 
CSV_PATH = BASE_DIR.parent / "data" / "processed" / "dataset_FULL.csv"

print("Lendo:", CSV_PATH)
df = pd.read_csv(CSV_PATH) # lendo
print("\n=========== Shape do dataset carregado: =============================================================================================================================")
print("Formato do DataFrame:", df.shape) # (N_linhas, N_colunas)
print("=========== Parâmetros de Entrada (X): ==============================================================================================================================")
print(df.columns[1:23].to_list())    # seed + 22 parâmetros, por exemplo
print("=====================================================================================================================================================================")

param_cols = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n", "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH", "dt", "t_end", "C_in", "T_in",
]
print("\n=========================================================================================================================")
print("Conferindo se todas as colunas de X existem no df:", all(col in df.columns for col in param_cols))
print("=========================================================================================================================")
out_final_cols = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]

all_cols = df.columns
Cz_cols   = [c for c in all_cols if c.startswith("C_z")] # JUNCAO DE TODOS OS Cz
Tz_cols   = [c for c in all_cols if c.startswith("T_z")] # JUNCAO DE TODOS OS Tz
Qtot_cols = [c for c in all_cols if c.startswith("Qtot_t")] # JUNCAO DE TODOS OS Qtot

output_cols = out_final_cols + Cz_cols + Tz_cols + Qtot_cols # JUNCAO DE TODAS AS COLUNAS DE Y

print("\n============= Colunas de saída (Y) – primeiras 10: =============================================================")
print(output_cols[:10])
print("=============== Total de saídas: ===============================================================================")
print(len(output_cols))
print("================================================================================================================")
# =======================================================
# DEFINICAO DO X e Y
X = df[param_cols].to_numpy(dtype=np.float32) # 
Y = df[output_cols].to_numpy(dtype=np.float32)

print("\n============ Shapes das matrizes: ==============")
print("X shape:", X.shape) # X shape: (93420, 22)
print("Y shape:", Y.shape) # Y shape: (93420, 157) (4 + 51 + 51 + 51, se forem 51 pontos em cada grupo)
print("================================================")
# =======================================================
scaler_X = StandardScaler().fit(X) # CALCULA A MEDIA E O DESVIO PADRAO DE CADA ENNTRADA
X = scaler_X.transform(X) # (X - μ) / σ X = entrada μ = média, σ = desvio padrão

# salvar o scaler pra usar depois na inferência
joblib.dump(scaler_X, BASE_DIR / "scaler_input.save") # guarda o scaler

# 5) Dimensões de entrada e saída
input_dim = X.shape[1] # número de características de entrada
output_dim = Y.shape[1] # 

print("\n=============Dimensões para a rede:=============")
print("input_dim:", input_dim)
print("output_dim:", output_dim)
print("================================================")

# def build_model(hp):
#     n1 = hp.Int('neurons_layer1', 128, 512, step=64)
#     n2 = hp.Int('neurons_layer2', 128, 512, step=64)
#     n3 = hp.Int('neurons_layer3', 64, 256, step=32)

#     dropout_rate = hp.Float('dropout', 0.0, 0.08, step=0.02)
#     l2_reg = hp.Choice('l2_reg', values=[1e-6, 1e-5, 1e-4])
#     lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

#     model = models.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(n1, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
#         layers.Dropout(dropout_rate),
#         layers.Dense(n2, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
#         layers.Dropout(dropout_rate),
#         layers.Dense(n3, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
#         layers.Dense(output_dim, activation='linear')
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#         metrics=[
#             tf.keras.metrics.MeanAbsoluteError(name='mae'),
#             tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
#             tf.keras.metrics.RootMeanSquaredError(name='rmse')
#         ],
#         loss='mse'
#     )
#     return model


# # =======================================================
# # 3. Otimização Bayesiana de hiperparâmetros
# # =======================================================
# tuner = kt.BayesianOptimization(
#     build_model,
#     objective='val_loss',
#     max_trials=10,
#     directory='tuning_logs',
#     project_name='adsorption_model_tuning'
# )

# tuner.search(X, Y, validation_split=0.2, epochs=200, batch_size=512, verbose=1)
# best_hp = tuner.get_best_hyperparameters(1)[0]
# model = build_model(best_hp)

# # =======================================================
# # 4. Treinamento final
# # =======================================================
# callbacks = [
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6),
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
# ]

# history = model.fit(
#     X, Y,
#     epochs=200,
#     batch_size=512,
#     validation_split=0.1,
#     callbacks=callbacks,
#     verbose=1
# )

# # =======================================================
# # 5. Impressão de métricas
# # =======================================================
# train_loss = history.history['loss'][-1]
# val_loss = history.history['val_loss'][-1]

# print("\n===== Resultados do Treinamento =====")
# print(f"Loss final (Treino): {train_loss:.6e}")
# print(f"Loss final (Validação): {val_loss:.6e}")

# # Cálculo de RMSE geral
# Y_pred = model.predict(X)
# rmse = np.sqrt(mean_squared_error(Y, Y_pred))
# print(f"RMSE Global (Treino): {rmse:.6f}")

# # =======================================================
# # 6. Gráfico de curva de treinamento
# # =======================================================
# plt.figure(figsize=(8,5))
# plt.plot(history.history['loss'], label='loss (treino)')
# plt.plot(history.history['val_loss'], label='loss (val)')
# plt.xlabel("Épocas")
# plt.ylabel("Loss (MSE)")
# plt.title("Curva de Treinamento")
# plt.legend()
# plt.grid(True)
# plt.savefig("curva_treinamento.png", dpi=300)
# plt.close()

# print("Gráfico salvo como curva_treinamento.png")

# # =======================================================
# # 7. Salvamento do modelo final
# # =======================================================
# model.save("modelo_adsorcao_filtrado.h5")
# print("Modelo salvo como modelo_adsorcao_filtrado.h5")