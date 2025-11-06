import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(42); tf.random.set_seed(42)

d = pd.read_csv("dataset_nu_Uvec_analitico.csv")
X = d[['nu']].to_numpy('float32')
y = d[[f'u_{k}' for k in range(25)]].to_numpy('float32')

Xtr, Xtmp, Ytr, Ytmp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
Xva, Xte, Yva, Yte = train_test_split(Xtmp, Ytmp, test_size=0.5, random_state=42, shuffle=True)

nu_min, nu_max = Xtr.min(), Xtr.max()
nrm = lambda a: (a - nu_min) / (nu_max - nu_min + 1e-12)
Xtr_n, Xva_n, Xte_n = nrm(Xtr), nrm(Xva), nrm(Xte)

y_mean, y_std = Ytr.mean(0, keepdims=True), Ytr.std(0, keepdims=True) + 1e-8
Ytr_n, Yva_n, Yte_n = (Ytr - y_mean)/y_std, (Yva - y_mean)/y_std, (Yte - y_mean)/y_std

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(25, activation="linear")
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
]

model.fit(Xtr_n, Ytr_n, validation_data=(Xva_n, Yva_n), epochs=2000, batch_size=64, callbacks=cbs, verbose=1)

mse, mae = model.evaluate(Xte_n, Yte_n, verbose=0)
print(f"Test MSE(norm): {mse:.6f} | MAE(norm): {mae:.6f}")

Ypred = model.predict(Xte_n, verbose=0)*y_std + y_mean
Ytrue = Yte

err = np.abs(Ypred - Ytrue)
rel = np.where(np.abs(Ytrue) > 1e-12, err/np.abs(Ytrue), 0.0)
MAE, MRE, MaxE = err.mean(), 100.0*rel.mean(), err.max()
R2 = 1.0 - np.sum((Ytrue - Ypred)**2)/np.sum((Ytrue - Ytrue.mean(0))**2)
print(f"MAE(real): {MAE:.6f} | MRE(real): {MRE:.3f}% | MaxE: {MaxE:.6f} | R²: {R2:.4f}")

TOL_ABS, TOL_REL = 1e-5, 5e-5   # TOL_REL=0.00005 (0,005%)
ok_abs = err <= TOL_ABS
rel_ok = np.where(np.abs(Ytrue) > 1e-12, err/np.abs(Ytrue) <= TOL_REL, ok_abs)
ok = np.logical_or(ok_abs, rel_ok)
acc_cells = ok.mean()
acc_sample = (ok.mean(1) >= 0.90).mean()
print(f"Acurácia por célula: {100*acc_cells:.2f}% | Acurácia por amostra (≥90% ok): {100*acc_sample:.2f}%")
