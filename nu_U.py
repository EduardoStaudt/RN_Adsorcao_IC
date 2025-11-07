#IMPORTS
import numpy as np, math
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras 
from sklearn.model_selection import train_test_split

#  CARREGGAR O .CSV
dataset = pd.read_csv("dataset_nu_Uvec_analitico.csv") # careregamento do csv(dataset)
print(dataset.head())# mostra as primeiras linhas do dataset

X = dataset[['nu']].to_numpy('float32') # transforma em uma coluna e depois em array numpy (entrada da rede)
y = dataset[[f'u_{k}' for k in range(25)]].to_numpy('float32') # cria uma lista com os nomes u_n e transforma em array numpy (saida da rede)

#SPLIT DATASET INTO TRAINING AND TEST SETS
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
    # 80% treino, 20% teste shuffle=embaralha os dados random_state=42 para reprodutibilidade mesma divisao toda vez o 42 dois nao faz nada so garante elle só guarda como o sorteio dos numeros foi feito(BEM ABSTRATO)
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    # Pega o 20% temporário e divide ao meio: 10% validação e 10% teste
)

#NORMALIZACAO DO DATASET
nu_min, nu_max, = X_train.min(), X_train.max()
X_train_n = (X_train - nu_min) / (nu_max - nu_min + 1e-12)  
X_validation_n = (X_validation - nu_min) / (nu_max - nu_min + 1e-12) 
X_test_n = (X_test - nu_min) / (nu_max - nu_min + 1e-12)
# calcula nu min e max só no conjunto de treino e depois normaliza e deixa [0,1] e o +1e-12 é para evitar divisão por zero caso nu_max==nu_min

y_mean = y_train.mean(axis=0, keepdims=True)
y_std = y_train.std(axis=0, keepdims=True) + 1e-8
y_train_n = (y_train - y_mean) / y_std  
y_validation_n = (y_validation - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std
# Calcula média e desvio-padrão por componente (25 saídas) no treino o keepdims=True mantém a dimensão para facilitar a subtração depois e o +1e-8 evita divisão por zero caso o desvio seja zero

print("\n--------------------------------\n")
print('Tabela de Entradas:\n' + str(X.shape) + '\nTabela de Saidas\n' + str(y.shape)) 
# verificação do formato dos arrays de entrada e saída

# CONSTRUÇÃO DO MODELO
model = tf.keras.Sequential([
    # geralmente a funcao relu nao é mais recomendada mas foi a que deu os melhores resultados nos testes
    # SWISH= x * sigmoid(x)
    # boa pois permite valores negativos de pequena magnitude (mínimo ≈ −0.278) mas de forma bem suave para caso ocorra multplicação por valores negativos e tranformar em positivo nao cause grandes erros
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
# mse= erro media quadratico mae= erro absoluto medio

# RESUMO DO MODELO
model.summary()

# CALLBACK(PARA AJUSTE DO LEARNING RATE E SALVAMENTO DO MELHOR MODELO)

# verbose=0 → silencioso (não imprime nada).

# verbose=1 → barra de progresso e métricas por época (ideal para Colab/VS Code).

# verbose=2 → uma linha por época (sem barra; útil quando a barra polui o log).

cbs = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    # observa a perda o val_loss e reduz o learning rate pela metade se nao melhorar por 10 epocas ate um minimo de 1e-6
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
    # salva o melhor modelo com base na perda de validação
]
history = model.fit(
    X_train_n, y_train_n,
    validation_data=(X_validation_n, y_validation_n), # Keras calcula val_loss/val_mae ao fim de cada época.
    epochs=500,                 
    batch_size=64, 
    callbacks=cbs,               
    verbose=1
)
#CARREGAR O MELHOR MODELO
model = tf.keras.models.load_model("best_model.keras")

#AVALIAÇÃO DO MODELO
# DESNORMALIZACAO
y_pred_n = model.predict(X_test_n, verbose=0)   # Gera Shapes: (N_test, 25)       
y_pred   = y_pred_n * y_std + y_mean # Desnormaliza as previsões para a escala real usando a média e o desvio do treino y_std e y_mean têm shape (1, 25)         
y_true   = y_test                                      

abs_err = np.abs(y_pred - y_true) # Erro absoluto distancia entre o valor verdadeiro e o previsto
with np.errstate(divide='ignore', invalid='ignore'):
    rel_err = np.where(np.abs(y_true) > 1e-12, abs_err/np.abs(y_true), 0.0)
    # erro relativo = erro absoluto / |valor verdadeiro|. Se o valor verdadeiro for muito próximo de zero (|valor verdadeiro| ≤ 1e-12), define o erro relativo como 0.0 para evitar divisão por zero ou valores muito grandes. Sai dividido por sem depois no calculo das metricas é multiplcado por 100

# CÁLCULO DAS MÉTRICAS DE DESEMPENHO
MAE  = abs_err.mean() # Erro absoluto médio
MRE  = 100.0 * rel_err.mean() # Erro relativo médio
MaxE = abs_err.max() # Máximo erro absoluto vai mostrar o pior caso do treinamento
den = np.sum((y_true - y_true.mean(axis=0))**2) + 1e-12 # Denominador para R² (com prevenção contra divisão por zero)
R2  = 1.0 - np.sum((y_true - y_pred)**2)/den # Coeficiente de determinação R² que indica a proporção da variância dos dados que é explicada pelo modelo

# ANÁLISE DE RESULTADOS E PRINT U_vec
np.set_printoptions(precision=6, suppress=True) 
# presisao de 6 casas decimais e suprimir a notação científica para facilitar a leitura

def show_vectors(i):
    nu_val = float(X_test[i, 0])  # pega o escalar nu dessa amostra.           
    u_true = y_true[i]     # vetor verdadeiro (shape (25,)) para essa amostra.                  
    u_pred = y_pred[i]      # vetor previsto (shape (25,)) para essa amostra.                 
    abs_err = np.abs(u_pred - u_true) # erro absoluto elemento a elemento

    # DataFrame para exibir os vetores lado a lado com verdadeiro vs. previsto e o erro absoluto
    comp = pd.DataFrame({
        "u_true": u_true,
        "u_pred": u_pred,
        "abs_err": abs_err
    }, index=[f"u_{k}" for k in range(25)])
    print(f"\n=== Amostra {i} | nu = {nu_val:.6f} ===")
    print(comp.to_string())

# MOSTRAR O PRIMEIRO U_vec()
print("\n--- PRIMEIRA AMOSTRA DE TESTE ---\n")
show_vectors(0)

# MOSTRAR AS 3 PIORES PREVISÕES()
print("\n--- TRÊS PIORES AMOSTRAS DE TESTE ---\n")
mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)   # (N_test,)
worst_idx = np.argsort(mae_per_sample)[-3:]
for i in worst_idx:
    show_vectors(int(i))

# MOSTRAR AS 3 MELHORES PREVISÕES()
print("\n--- TRÊS MELHORES AMOSTRAS DE TESTE ---\n")
best_idx = np.argsort(mae_per_sample)[:3]
for i in best_idx:
    show_vectors(int(i))

# ANÁLISE DE ACURÁCIA COM BASE EM TOLERÂNCIAS
TOL_ABS = 0.00001      # tolerancia para erro absoluto 1e-5
TOL_REL = 0.00005      # tolerancia para erro relativo 5e-5
SAMPLE_FRAC_OK = 0.90  # fração mínima de células "ok" por amostra

ok_abs = abs_err <= TOL_ABS # verifica se o erro absoluto está dentro da tolerância
with np.errstate(divide='ignore', invalid='ignore'): #np.errstate(...) só silencia warnings de divisão por zero/NaN dentro do bloco.
    rel_ok = np.where(np.abs(y_true) > 1e-12, abs_err/np.abs(y_true) <= TOL_REL, ok_abs)  # verifica se o erro relativo está dentro da tolerância
ok = np.logical_or(ok_abs, rel_ok)  # A célula é considerada OK se passar em pelo menos um dos critérios (absoluto OU relativo).

acc_cells = ok.mean() # porcentagem de células ok em todas as amostras e todas as 25 saídas.

acc_per_sample = (ok.mean(axis=1) >= SAMPLE_FRAC_OK).mean() # porcentagem de amostras que têm pelo menos SAMPLE_FRAC_OK (90%) de suas células ok.

print("\n----------------MÉTRICAS DE DESEMPENHO----------------\n")
print(f'Test MAE (real): {MAE:.6f}  \nTest MRE (real): {MRE:.3f}%  \nTest MaxE (real): {MaxE:.6f}  \nR²: {R2:.4f}')

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
    # Acurácia por célula (tolerância): 84.50%
    # Acurácia por amostra (≥90% células ok): 67.20%
#com 10000 epochs
    # Acurácia por célula (tolerância): 100.00%
    # Acurácia por amostra (≥90% células ok): 100.00%

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