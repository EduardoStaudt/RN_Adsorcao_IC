#IMPORTS
import numpy as np, math
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras # se tira o .keras o codigo nao funciona
from sklearn.model_selection import train_test_split

#  CARREGGAR O .CSV
dataset = pd.read_csv("dataset_nu_Uvec_analitico.csv") # careregamento do csv(dataset)
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

# CONSTRUÇÃO DO MODELO
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="relu", input_shape=(1,), use_bias=True),
    tf.keras.layers.Dense(25, activation="linear", use_bias=True)   
    # tf.keras.layers.Dense(25, activation="linear", input_shape=(1,), use_bias=True)
    # tf.keras.layers.Input(shape=(1,)),
    # tf.keras.layers.Dense(64, activation="swish",
    #                     kernel_initializer="he_uniform",
    #                     kernel_regularizer=keras.regularizers.l2(1e-5)),
    # tf.keras.layers.Dense(64, activation="tanh",
    #                     kernel_initializer="he_uniform",
    #                     kernel_regularizer=keras.regularizers.l2(1e-5)),
    # tf.keras.layers.Dense(25, activation="linear")
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

# RESUMO DO MODELO
model.summary()
# CALLBACK(PARA AJUSTE DO LEARNING RATE E SALVAMENTO DO MELHOR MODELO)
cbs = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
]
history = model.fit(
    X_train_n, y_train_n,
    validation_data=(X_validation_n, y_validation_n),
    epochs=2000,                 # alto, mas com early stopping
    batch_size=64, 
    callbacks=cbs,               # menor ajuda a generalizar
    verbose=1
)
#CARREGAR O MELHOR MODELO
model = tf.keras.models.load_model("best_model.keras")

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
# R2   = 1.0 - np.sum((y_true - y_pred)**2)/np.sum((y_true - y_true.mean(axis=0))**2)
den = np.sum((y_true - y_true.mean(axis=0))**2) + 1e-12
R2  = 1.0 - np.sum((y_true - y_pred)**2)/den

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

# ANÁLISE DE ACURÁCIA COM BASE EM TOLERÂNCIAS
TOL_ABS = 0.00001   # ajuste conforme a escala do seu U (ex.: 0.01 = 1e-2)
TOL_REL = 0.00005   # % de erro relativo
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


# MELHOR MODELO DE TODOS OS TESTADOS
    # tf.keras.layers.Dense(1, activation="relu", input_shape=(1,), use_bias=True),
    # tf.keras.layers.Dense(25, activation="linear", use_bias=True)  
#com 5000 epochs
# Acurácia por célula (tolerância): 99.60%
# Acurácia por amostra (≥90% células ok): 99.50%
#com 2000 epochs





#RESULTADOS APÓS EDIÇÕES

# TOL_ABS = 0.00001   
# TOL_REL = 0.00005  
#CAMADA DUPLA
# Acurácia por célula (tolerância): 80.62%
# Acurácia por amostra (≥90% células ok): 71.50%
# CAMADA SIMPLES
# Acurácia por célula (tolerância): 71.60%
# Acurácia por amostra (≥90% células ok): 0.00%

# NOVO MODELO COM 3 CAMADAS 
# TOL_ABS = 0.00001   
# TOL_REL = 0.00005  
# SWISH:
# Acurácia por célula (tolerância): 20.52%
# Acurácia por amostra (≥90% células ok): 0.00%
# TANH:
# Acurácia por célula (tolerância): 20.66%
# Acurácia por amostra (≥90% células ok): 0.00%
# SWISH AND TANH:
# Acurácia por célula (tolerância): 20.78%
# Acurácia por amostra (≥90% células ok): 0.00%



# FIM DOS RESULTADOS APÓS EDIÇÕES



























# PLOT CURVA DE TREINO()
# plt.figure()
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="val")
# plt.xlabel("Epoch"); plt.ylabel("MSE (norm)"); plt.legend(); plt.title("Curva de treino")
# plt.tight_layout(); plt.show()

# PLOT CURVA DE TREINO()
# plt.figure()
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="val")
# plt.xlabel("Epoch"); plt.ylabel("MSE (norm)"); plt.legend(); plt.title("Curva de treino")
# plt.tight_layout(); plt.show()

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