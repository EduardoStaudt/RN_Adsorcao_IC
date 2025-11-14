import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import keras_tuner as kt
import joblib
import matplotlib.pyplot as plt
# =======================================================
# 1. Carregamento do dataset
# =======================================================
data = np.load("dataset_treino_filtrado.npz")

print("Chaves disponíveis:", list(data.keys()))
X = data["X_train"]
Y = data["y_train"]

# Normalização
scaler_X = StandardScaler().fit(X)
X = scaler_X.transform(X)
joblib.dump(scaler_X, "scaler_input.save")

input_dim = X.shape[1]
output_dim = Y.shape[1]

# =======================================================
# 2. Função para criação do modelo (para tuning)
# =======================================================
def build_model(hp):
    n1 = hp.Int('neurons_layer1', 128, 512, step=64)
    n2 = hp.Int('neurons_layer2', 128, 512, step=64)
    n3 = hp.Int('neurons_layer3', 64, 256, step=32)

    dropout_rate = hp.Float('dropout', 0.0, 0.08, step=0.02)
    l2_reg = hp.Choice('l2_reg', values=[1e-6, 1e-5, 1e-4])
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n1, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(n2, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(n3, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(output_dim, activation='linear')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ],
        loss='mse'
    )
    return model


# =======================================================
# 3. Otimização Bayesiana de hiperparâmetros
# =======================================================
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,
    directory='tuning_logs',
    project_name='adsorption_model_tuning'
)

tuner.search(X, Y, validation_split=0.2, epochs=200, batch_size=512, verbose=1)
best_hp = tuner.get_best_hyperparameters(1)[0]
model = build_model(best_hp)

# =======================================================
# 4. Treinamento final
# =======================================================
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
]

history = model.fit(
    X, Y,
    epochs=200,
    batch_size=512,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# =======================================================
# 5. Impressão de métricas
# =======================================================
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print("\n===== Resultados do Treinamento =====")
print(f"Loss final (Treino): {train_loss:.6e}")
print(f"Loss final (Validação): {val_loss:.6e}")

# Cálculo de RMSE geral
Y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print(f"RMSE Global (Treino): {rmse:.6f}")

# =======================================================
# 6. Gráfico de curva de treinamento
# =======================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='loss (treino)')
plt.plot(history.history['val_loss'], label='loss (val)')
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.title("Curva de Treinamento")
plt.legend()
plt.grid(True)
plt.savefig("curva_treinamento.png", dpi=300)
plt.close()

print("Gráfico salvo como curva_treinamento.png")

# =======================================================
# 7. Salvamento do modelo final
# =======================================================
model.save("modelo_adsorcao_filtrado.h5")
print("Modelo salvo como modelo_adsorcao_ajustado.h5")