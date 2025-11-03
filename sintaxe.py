# ============================================================
# GUIA COMPLETO DE SINTAXE TENSORFLOW – REDE NEURAL EXPLICADA
# ============================================================

# 1️⃣ Importando as bibliotecas principais
import tensorflow as tf                 # TensorFlow → principal biblioteca de Deep Learning
from tensorflow import keras             # keras → API de alto nível para redes neurais
from tensorflow.keras import layers      # layers (camadas) → blocos fundamentais da rede neural

# 2️⃣ Preparando os dados (dataset)
# Aqui usamos o MNIST (imagens de dígitos de 0 a 9) como exemplo padrão.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizamos (normalization) → escalamos os valores para o intervalo [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3️⃣ Construindo o modelo (model)
# Sequential (sequencial) → modelo onde as camadas são empilhadas linearmente.
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),         # Flatten (achatamento): transforma a imagem 2D em vetor 1D
    layers.Dense(128, activation='relu'),         # Dense (camada totalmente conectada)
    # activation (ativação): função que decide a saída do neurônio.
    # relu (Rectified Linear Unit): zera valores negativos, acelera aprendizado.

    layers.Dropout(0.2),                          # Dropout (desligamento aleatório): previne overfitting (sobreajuste)

    layers.Dense(10, activation='softmax')        # softmax: converte as saídas em probabilidades (somam 1)
])

# 4️⃣ Compilando o modelo (compile)
# Define como o modelo será treinado (otimizador, função de perda e métricas)
model.compile(
    optimizer='adam',              # optimizer (otimizador): ajusta os pesos — Adam é rápido e eficiente
    loss='sparse_categorical_crossentropy',  # loss (perda): mede o erro entre previsão e rótulo
    metrics=['accuracy']           # metrics (métricas): avalia o desempenho do modelo
)

# 5️⃣ Treinando o modelo (fit)
# fit (ajustar): inicia o processo de aprendizado
history = model.fit(
    x_train, y_train,              # dados de entrada e saída (features e labels)
    epochs=5,                      # epochs (épocas): quantas vezes o modelo verá todo o dataset
    batch_size=32,                 # batch_size (tamanho do lote): nº de amostras antes de atualizar os pesos
    validation_split=0.1,          # separa 10% dos dados para validação automática
    verbose=1                      # verbose (detalhamento): 1 mostra barra de progresso
)

# 6️⃣ Avaliando o modelo (evaluate)
# evaluate (avaliar): mede o desempenho nos dados de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia (accuracy) no teste: {test_acc:.3f}")

# 7️⃣ Fazendo previsões (predict)
# predict (prever): gera saídas para novos dados
predictions = model.predict(x_test[:5])

# Exemplo: exibindo a classe com maior probabilidade
predicted_labels = tf.argmax(predictions, axis=1)
print("Rótulos previstos:", predicted_labels.numpy())
print("Rótulos reais:", y_test[:5])

# ============================================================
# 🧠 CONCEITOS-CHAVE (tradução e função)
# ============================================================
# - model (modelo): estrutura que contém camadas, pesos e lógica de aprendizado
# - layer (camada): unidade da rede; processa entradas e passa saídas
# - activation (ativação): define como o neurônio reage à entrada (não-linearidade)
# - optimizer (otimizador): atualiza pesos para reduzir a perda
# - loss (função de perda): mede o erro entre saída prevista e real
# - epoch (época): uma passagem completa pelos dados
# - batch (lote): subconjunto dos dados em cada atualização
# - fit (ajustar): processo de treino
# - evaluate (avaliar): mede desempenho do modelo
# - predict (prever): usa o modelo treinado para inferir novas saídas
# ============================================================

# 8️⃣ Salvando e carregando o modelo (save / load)
# save (salvar) → cria um arquivo com os pesos e arquitetura
model.save("meu_modelo.h5")

# load_model (carregar) → reabre o modelo salvo
modelo_carregado = keras.models.load_model("meu_modelo.h5")

# Confirmando que o modelo foi restaurado corretamente
print("Modelo carregado com sucesso!")