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

# ┌──────────── BLOCO DE SAÍDAS FINAIS (resumo da simulação) ─────────────┐
# │ C_out_final   → concentração na SAÍDA no tempo final                  │
# │ q_out_final   → carga adsorvida (na saída/média) no tempo final       │
# │ T_out_final   → temperatura na SAÍDA no tempo final                   │
# │ N_ads_final   → quantidade total adsorvida no leito (integral de q)   │
# └───────────────────────────────────────────────────────────────────────┘

all_cols = df.columns
Cz_cols   = [c for c in all_cols if c.startswith("C_z")] # JUNCAO DE TODOS OS Cz
Tz_cols   = [c for c in all_cols if c.startswith("T_z")] # JUNCAO DE TODOS OS Tz
Qtot_cols = [c for c in all_cols if c.startswith("Qtot_t")] # JUNCAO DE TODOS OS Qtot

output_cols = out_final_cols + Cz_cols + Tz_cols + Qtot_cols # JUNCAO DE TODAS AS COLUNAS DE Y

n_final = len(out_final_cols)   # 4
n_Cz    = len(Cz_cols)          # 51
n_Tz    = len(Tz_cols)          # 51
n_Qtot  = len(Qtot_cols)        # 51


print("\n============= Colunas de saída (Y) – primeiras 10: =============================================================")
print(output_cols[:10])
print("=============== Total de saídas: ===============================================================================")
print(len(output_cols))
print("================================================================================================================")
# =======================================================
# DEFINICAO DO X e Y 
# NOTE:utilizei o TFT 
X = df[param_cols].to_numpy(dtype=np.float32) # 
Y = df[output_cols].to_numpy(dtype=np.float32)

print("\n============ Shapes das matrizes: ==============")
print("X shape:", X.shape) # X shape: (93420, 22)
print("Y shape:", Y.shape) # Y shape: (93420, 157) (4 + 51 + 51 + 51, se forem 51 pontos em cada grupo)
print("================================================")

# APLICANDO O Z-score
scaler_X = StandardScaler().fit(X) # CALCULA A MEDIA E O DESVIO PADRAO DE CADA COLUNA
X = scaler_X.transform(X) # (X - μ) / σ X = entrada μ = média, σ = desvio padrão

scaler_Y = StandardScaler().fit(Y) # CALCULA A MEDIA E O DESVIO PADRAO DE CADA COLUNA
Y = scaler_Y.transform(Y) # (Y - μ) / σ Y = saída μ = média, σ = desvio padrão

# salvar o scaler pra usar depois na inferência
joblib.dump(scaler_X, BASE_DIR / "scaler_input.save") # guarda o scaler
joblib.dump(scaler_Y, BASE_DIR / "scaler_output.save") # guarda o scaler


#SALVAR DIMENSOES
input_dim = X.shape[1] # número de características de entrada
output_dim = Y.shape[1] # 

# VERIFICACAO DA NORMALIZACAO 
print("\nChecando normalização de X (primeiras 5 features):")
print("médias  :", X.mean(axis=0)[:5])
print("desvios :", X.std(axis=0)[:5])

print("\nChecando normalização de Y (primeiras 5 saídas):")
print("médias  :", Y.mean(axis=0)[:5])
print("desvios :", Y.std(axis=0)[:5])

print("\n=============Dimensões para a rede:=============")
print("input_dim:", input_dim)
print("output_dim:", output_dim)
print("================================================")

W_FINAL = 4.0   # dê mais importância às 4 saídas finais
W_CZ    = 1.0
W_TZ    = 1.0
W_QTOT  = 1.0

def weighted_mse(y_true, y_pred):
    """
    MSE ponderado por grupos de saídas:
    - 4 finais (C_out, q_out, T_out, N_ads)
    - C_z*
    - T_z*
    - Qtot_t*
    """

    err2 = tf.square(y_true - y_pred)  # (batch, 157)

    # fatias por grupo
    e_final = err2[:, :n_final]
    e_Cz    = err2[:, n_final:n_final + n_Cz]
    e_Tz    = err2[:, n_final + n_Cz : n_final + n_Cz + n_Tz]
    e_Qtot  = err2[:, n_final + n_Cz + n_Tz : ]

    # MSE de cada grupo (média em batch + saídas)
    mse_final = tf.reduce_mean(e_final)
    mse_Cz    = tf.reduce_mean(e_Cz)
    mse_Tz    = tf.reduce_mean(e_Tz)
    mse_Qtot  = tf.reduce_mean(e_Qtot)

    # Combinação ponderada
    return (
        W_FINAL * mse_final +
        W_CZ    * mse_Cz +
        W_TZ    * mse_Tz +
        W_QTOT  * mse_Qtot
    )

def NeuralNetwork(hp):
    N1 = hp.Int('NeuralNetwork1', 88, 528, step=44)
    N2 = hp.Int('NeuralNetwork2', 176, 528, step=44)
    N3 = hp.Int('NeuralNetwork3', 88, 264, step=22)

    Drop = hp.Float('Dropout_Rate', 0.0, 0.3, step=0.05)
    l2_reg = hp.Choice('l2_reg', values=[1e-6, 1e-5, 1e-4])
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model = models.Sequential([
        layers.Input(shape=(input_dim,)), # 22 ENTRADAS
        layers.Dense(N1, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(Drop),
        layers.Dense(N2, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(Drop),
        layers.Dense(N3, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(output_dim, activation='linear') # 157 SAÍDAS
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), # COM TURNER
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'), 
            # erro absoluto médio: |y_true - y_pred|
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
            # Raiz do erro quadrático médio: sqrt(mean((y_true - y_pred)^2))
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
            # tf.keras.metrics.MeanSquaredError(name='mse')
        ],
        loss=weighted_mse, # DIMINUIR O MSE DURANTE O TREINAMENTO
    )
    return model


# =======================================================
# 3. Otimização Bayesiana de hiperparâmetros
# =======================================================
tuner = kt.BayesianOptimization(
    NeuralNetwork, # FUNÇÃO DE CONSTRUÇÃO DO MODELO
    objective='val_loss', # OBJETIVO MINIMIZAR O LOSS
    max_trials= 30, # NÚMERO DE HIPERPARÂMETROS TESTADOS
    directory='tuning_logs', # DIRETÓRIO ONDE SERÃO SALVOS OS LOGS
    project_name='adsorption_model_tuning', # NOME DO PROJETO PARA ORGANIZAÇÃO DOS LOGS
)

tuner.search(X, Y, validation_split=0.2, epochs=200, batch_size=512, verbose=0) 
best_hp = tuner.get_best_hyperparameters(1)[0]# COLETA DOS MELHORES HIPERPARAMETROS

model = NeuralNetwork(best_hp) # CONSTRUÇÃO DO MODELO COM OS MELHORES HIPERPARÂMETROS

# =======================================================
# 4. Treinamento final
# =======================================================

# Caminho pra salvar o melhor modelo
MODELS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

ckpt_path = MODELS_DIR / "best_model.keras"

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(ckpt_path),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

callbacks = [
    checkpoint_cb,
    #  REDUZ PELA METADE O LEARNING RATE SE A VAL_LOSS NÃO MELHORAR EM 12 ÉPOCAS
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6),
    # PARA O TREINAMENTO SE A VAL_LOSS NÃO MELHORAR EM 30 ÉPOCAS
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
]
# loss ↓ e val_loss ↓ - aprendendo bem.

# loss ↓ mas val_loss ↑ - a rede começou a decorar o treino (overfitting).

history = model.fit(
    X, Y,
    epochs=2000,
    batch_size=512,
    validation_split=0.1,
    callbacks=callbacks,
)

# =======================================================
# 5. Impressão de métricas
# =======================================================
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print("\n===== Resultados do Treinamento =====")
print(f"Loss final (Treino, espaço normalizado): {train_loss:.6e}")
print(f"Loss final (Validação, espaço normalizado): {val_loss:.6e}")
print("=====================================")
# ---------- 5.1 Predição no espaço normalizado ----------
Y_pred_norm = model.predict(X)  # modelo foi treinado com X e Y normalizados

rmse_norm = np.sqrt(mean_squared_error(Y, Y_pred_norm))
print(f"\nRMSE Global (Treino, espaço normalizado): {rmse_norm:.6f}")

# ---------- 5.2 Desnormalizar para espaço físico ----------
Y_real      = scaler_Y.inverse_transform(Y) # volta Y 
Y_pred_real = scaler_Y.inverse_transform(Y_pred_norm) # volta Y_pred

# RMSE DE TODOS AS 157 SAIDAS
rmse_real_global = np.sqrt(mean_squared_error(Y_real, Y_pred_real))
print(f"RMSE Global (Treino, espaço físico, todas as saídas): {rmse_real_global:.6f}")

# RMSEs ESPECIFICOS
#.index() retorna o índice da coluna especificada
# output_cols = [C_out_final, q_out_final, T_out_final, N_ads_final, C_z0... , T_z0..., Qtot_t0...]
idx_C_out = output_cols.index("C_out_final") # 23 0
idx_T_out = output_cols.index("T_out_final") # 25 2
idx_N_ads = output_cols.index("N_ads_final") # 26 3

# CALCULO DO MSE PARA CADA SAÍDA ESPECÍFICA REAL E PREDITA
rmse_C_out = np.sqrt(mean_squared_error(Y_real[:, idx_C_out], Y_pred_real[:, idx_C_out])) 
rmse_T_out = np.sqrt(mean_squared_error(Y_real[:, idx_T_out], Y_pred_real[:, idx_T_out]))
rmse_N_ads = np.sqrt(mean_squared_error(Y_real[:, idx_N_ads], Y_pred_real[:, idx_N_ads]))

# PRINTs
print("\n============RMSEs em espaço físico (saídas finais): ==========")
print(f"C_out_final  (concentração na saída): {rmse_C_out:.6f}")
print(f"T_out_final  (temperatura na saída):  {rmse_T_out:.6f}")
print(f"N_ads_final  (adsorção total):       {rmse_N_ads:.6f}")
print("=============================================================")
# =======================================================
# 6. Gráfico de curva de treinamento
# =======================================================

plt.figure(figsize=(8, 5)) # Cria uma nova figura (janela de desenho) com tamanho 8x5 polegadas

# Plota a curva da loss de treino ao longo das épocas
# history.history['loss'] é uma lista: [loss_epoch_1, loss_epoch_2, ...]
plt.plot(history.history['loss'], label='loss (treino)')

plt.plot(history.history['val_loss'], label='loss (val)') # Plota a curva da loss de validação ao longo das épocas

plt.xlabel("Épocas") # Rótulo do eixo X → estamos usando o índice da época (0,1,2,...) como X

plt.ylabel("Loss (MSE)") # Rótulo do eixo Y → valor da função de perda

plt.title("Curva de Treinamento") # Título do gráfico

plt.grid(True) # Adiciona uma grade (linhas de fundo) pra facilitar a leitura do gráfico

best_epoch = np.argmin(history.history['val_loss']) # Encontra a época com menor val_loss

plt.axvline(best_epoch, color='gray', linestyle='--', label=f'best epoch = {best_epoch}') # Marca a época de menor val_loss com uma linha vertical tracejada

plt.legend() # Mostra a legenda com os nomes definidos em 'label='

plt.savefig("curva_treinamento.png", dpi=300) # Salva a figura em um arquivo PNG com resolução de 300 dpi
print("Gráfico de curva de treinamento salvo como curva_treinamento.png")
plt.close() # Fecha a figura em memória (libera recursos, útil quando gera vários gráficos)

# =======================================================
# 7. Salvamento do modelo final
# =======================================================
print(f"Melhor modelo salvo em: {ckpt_path}")