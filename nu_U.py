#IMPORTS
import numpy as np, math
import scipy.sparse as sp, scipy.sparse.linalg as spla
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as keras
from sklearn.model_selection import train_test_split

#  CARREGGAR O .CSV
dataset = pd.read_csv("dataset_nu_Uvec_step0p002_full.csv") # careregamento do csv(dataset)
print(dataset.head())# amostra as primeiras linhas do dataset

X = dataset[['nu']].to_numpy('float32') # entradas da rede
y = dataset[[f'u_{k}' for k in range(25)]].to_numpy('float32') # saídas da rede

#SPLIT DATASET INTO TRAINING AND TEST SETS
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

#NORMALIZACAO DO DATASET
nu_min, nu_max, = X_train.min(), X_train.max()
X_train_n = (X_train - nu_min) / (nu_max - nu_min + 1e-12)  # normalização dos dados de entrada para o intervalo [0, 1]
X_validation_n = (X_validation - nu_min) / (nu_max - nu_min + 1e-12) 
X_test_n = (X_test - nu_min) / (nu_max - nu_min + 1e-12)

y_mean = y_train.mean(axis=0, keepdims=True)
y_std = y_train.std(axis=0, keepdims=True) + 1e-8
y_train_n = (y_train - y_mean) / y_std  # normalização dos dados de saída para média 0 e desvio padrão 1
y_validation_n = (y_validation - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

print("\n--------------------------------\n")
print('Tabela de Entradas:\n' + str(X.shape) + '\nTabela de Saidas\n' + str(y.shape)) # verificação do formato dos arrays de entrada e saída

#TREINAMENTO DO MODELO
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=5, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=25, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=125, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=25, activation= 'linear'),
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# RESUMO DO MODELO
model.summary()

# CALLBACKS(para evitar treinar atoa)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
]

history = model.fit(
    X_train_n, y_train_n,
    validation_data=(X_validation_n, y_validation_n),
    epochs=1000,
    batch_size=128,
    verbose=1
)

#AVALIAÇÃO DO MODELO
test_loss, test_mae = model.evaluate(X_test_n, y_test_n, verbose=0)
print('\nAvaliação do modelo no conjunto de teste:')
print(f'Test MSE (normalizado): {test_loss:.6f} | Test MAE (normalizado): {test_mae:.6f}')

# DESNORMALIZACAO
y_pred_n = model.predict(X_test_n, verbose=0)          # (M,25) normalizado
y_pred   = y_pred_n * y_std + y_mean                   # volta à escala real
y_true   = y_test                                      # já está na escala real

abs_err = np.abs(y_pred - y_true)
with np.errstate(divide='ignore', invalid='ignore'):
    rel_err = np.where(np.abs(y_true) > 1e-12, abs_err/np.abs(y_true), 0.0)

# CÁLCULO DAS MÉTRICAS DE DESEMPENHO
MAE  = abs_err.mean()
MRE  = 100.0 * rel_err.mean()
MaxE = abs_err.max()
R2   = 1.0 - np.sum((y_true - y_pred)**2)/np.sum((y_true - y_true.mean(axis=0))**2)

print(f'Test MAE (real): {MAE:.6f}  \nTest MRE (real): {MRE:.3f}%  \nTest MaxE (real): {MaxE:.6f}  \nR²: {R2:.4f}')

# ANÁLISE DE RESULTADOS E PRINT U_vec
np.set_printoptions(precision=6, suppress=True)

def show_vectors(i):
    """Imprime U_vec true vs pred + erro absoluto, e plota em 5x5."""
    nu_val = float(X_test[i, 0])             # nu no espaço real
    u_true = y_true[i]                       # shape (25,)
    u_pred = y_pred[i]                       # shape (25,)
    abs_err = np.abs(u_pred - u_true)

    comp = pd.DataFrame({
        "u_true": u_true,
        "u_pred": u_pred,
        "abs_err": abs_err
    }, index=[f"u_{k}" for k in range(25)])
    print(f"\n=== Amostra {i} | nu = {nu_val:.6f} ===")
    print(comp.to_string())

    # PLOTAGEM DOS MAPAS 5x5()
    # mapas 5x5
    # t5 = u_true.reshape(5,5)
    # p5 = u_pred.reshape(5,5)
    # e5 = (p5 - t5)

    # fig, axs = plt.subplots(1,3, figsize=(10,3))
    # im0 = axs[0].imshow(t5, origin="lower"); axs[0].set_title("True");  plt.colorbar(im0, ax=axs[0])
    # im1 = axs[1].imshow(p5, origin="lower"); axs[1].set_title("Pred");  plt.colorbar(im1, ax=axs[1])
    # im2 = axs[2].imshow(e5, origin="lower", cmap="seismic"); axs[2].set_title("Error"); plt.colorbar(im2, ax=axs[2])
    # plt.suptitle(f"nu={nu_val:.6f} — True vs Pred vs Error")
    # plt.tight_layout(); plt.show()

# MOSTRAR O PRIMEIRO U_vec()
show_vectors(0)

# MOSTRAR AS 3 PIORES PREVISÕES()
mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)   # (N_test,)
worst_idx = np.argsort(mae_per_sample)[-3:]
for i in worst_idx:
    show_vectors(int(i))

# MOSTRAR AS 3 MELHORES PREVISÕES()
best_idx = np.argsort(mae_per_sample)[:3]
for i in best_idx:
    show_vectors(int(i))

# PLOT CURVA DE TREINO()
# plt.figure()
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="val")
# plt.xlabel("Epoch"); plt.ylabel("MSE (norm)"); plt.legend(); plt.title("Curva de treino")
# plt.tight_layout(); plt.show()

# ANÁLISE DE ACURÁCIA COM BASE EM TOLERÂNCIAS
TOL_ABS = 0.01   # ajuste conforme a escala do seu U (ex.: 0.01 = 1e-2)
TOL_REL = 0.05   # 5% de erro relativo
SAMPLE_FRAC_OK = 0.90  # amostra é "correta" se >=90% das 25 células estão dentro da tolerância

ok_abs = abs_err <= TOL_ABS
with np.errstate(divide='ignore', invalid='ignore'):
    rel_ok = np.where(np.abs(y_true) > 1e-12, abs_err/np.abs(y_true) <= TOL_REL, ok_abs)  # se y_true≈0, use critério absoluto
ok = np.logical_or(ok_abs, rel_ok)  # (N, 25)

# Acurácia global por célula (todas as posições, todas as amostras)
acc_cells = ok.mean()  # em [0,1]
# Acurácia por amostra: fração de amostras com >= SAMPLE_FRAC_OK células corretas
acc_per_sample = (ok.mean(axis=1) >= SAMPLE_FRAC_OK).mean()

print("\n----------------ANALISE DE TOLERANCIAS----------------\n")
print(f"Acurácia por célula (tolerância): {100 * acc_cells:.2f}%")
print(f"Acurácia por amostra (≥{int(SAMPLE_FRAC_OK * 100)}% células ok): {100 * acc_per_sample:.2f}%")
