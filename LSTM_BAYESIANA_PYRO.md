# Modelo Bayesiano LSTM para Predicción de Precios de Activos Financieros

Este documento presenta el esbozo inicial de un modelo **Bayesiano LSTM** para la predicción de precios de activos financieros. El modelo utiliza **inferencia bayesiana** para proporcionar no solo predicciones puntuales, sino también estimaciones de incertidumbre, lo que resulta crucial para la toma de decisiones financieras. Además, la solución está optimizada para **computación paralela**, permitiendo un entrenamiento y predicciones más eficientes.

---

## Características principales

- **Inferencia Bayesiana**: Utiliza Pyro para implementar inferencia variacional estocástica (SVI).  
- **Arquitectura LSTM**: Modelo de redes neuronales recurrentes para capturar dependencias temporales en los datos históricos de precios.  
- **Procesamiento paralelo**: Implementación optimizada para aprovechar múltiples núcleos de la CPU.  
- **Cuantificación de incertidumbre**: Proporciona intervalos de confianza para las predicciones, útiles en la toma de decisiones financieras.  
- **Soporte para GPU**: Aceleración mediante CUDA cuando está disponible.  
- **Visualización avanzada**: Gráficos intuitivos de predicciones con bandas de incertidumbre.

---

## Código completo

A continuación se presenta el código principal. Se ha organizado en pasos claros, desde la descarga de datos hasta la inferencia bayesiana y la visualización de resultados:

```python
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
from functools import partial

# 1️⃣ Descargar datos de precios de acciones
def get_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    """
    Descarga datos históricos de precios de acciones usando yfinance.
    
    Args:
        ticker: Símbolo de la acción
        start: Fecha de inicio (formato YYYY-MM-DD)
        end:   Fecha de fin (formato YYYY-MM-DD)
    
    Returns:
        Array numpy con precios de cierre
    """
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError(f"No se encontraron datos para {ticker}")
        return df[["Close"]].values  # Solo precio de cierre
    except Exception as e:
        print(f"Error al descargar datos: {e}")
        raise

# 2️⃣ Preprocesar datos
def prepare_data(data, seq_length=30, test_size=0.2):
    """
    Preprocesa los datos para el entrenamiento del modelo LSTM.
    
    Args:
        data: Array de precios
        seq_length: Longitud de la secuencia para predecir
        test_size: Proporción de datos para prueba
    
    Returns:
        Tensores de entrenamiento/prueba y scaler para desnormalizar
    """
    # Normalizar datos entre 0 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Crear secuencias X e Y
    x_data, y_data = [], []
    for i in range(len(data_scaled) - seq_length):
        x_data.append(data_scaled[i:i + seq_length])
        y_data.append(data_scaled[i + seq_length])

    x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32)
    
    # Dividir en conjuntos de entrenamiento y prueba
    train_size = int(len(x_tensor) * (1 - test_size))
    x_train, x_test = x_tensor[:train_size], x_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    return (x_train, y_train), (x_test, y_test), scaler

# 3️⃣ Definir el modelo LSTM
class BayesianLSTM(nn.Module):
    """
    Modelo LSTM para predicción de series temporales.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Añadir dropout para regularización
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inicializar estados ocultos
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Última salida
        return out

# 4️⃣ Modelo probabilístico en Pyro
def pyro_model(x, y=None, model=None):
    """
    Define el modelo probabilístico para inferencia bayesiana.
    
    Args:
        x: Datos de entrada
        y: Datos objetivo (opcional)
        model: Modelo LSTM
    """
    if model is None:
        raise ValueError("Se debe proporcionar un modelo")
    
    # Priors para pesos y sesgos
    w_prior = dist.Normal(torch.zeros_like(model.fc.weight), torch.ones_like(model.fc.weight))
    b_prior = dist.Normal(torch.zeros_like(model.fc.bias), torch.ones_like(model.fc.bias))
    
    # Diccionario de priors
    priors = {"fc.weight": w_prior, "fc.bias": b_prior}

    # Levantar el módulo con los priors
    lifted_module = pyro.random_module("module", model, priors)()
    
    # Predicción del modelo
    y_hat = lifted_module(x)

    # Prior para la varianza del ruido
    sigma = pyro.sample("sigma", dist.Uniform(0.01, 0.5))
    
    # Likelihood
    with pyro.plate("data", x.size(0)):
        pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)

    return y_hat

# 5️⃣ Entrenar con inferencia bayesiana
def train(model, x_train, y_train, num_epochs=1000, batch_size=64, lr=0.01):
    """
    Entrena el modelo usando SVI (Stochastic Variational Inference).
    
    Args:
        model: Modelo LSTM
        x_train, y_train: Datos de entrenamiento
        num_epochs: Número de épocas
        batch_size: Tamaño del lote
        lr: Tasa de aprendizaje
    
    Returns:
        Lista de pérdidas durante el entrenamiento y la guía entrenada
    """
    import pyro.infer
    # Limpiar parámetros anteriores
    pyro.clear_param_store()
    
    # Crear dataset y dataloader para procesamiento por lotes
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Configurar guía y optimizador
    guide = pyro.infer.autoguide.AutoNormal(lambda x, y: pyro_model(x, y, model))
    optimizer = Adam({"lr": lr})
    svi = SVI(
        model=lambda x, y: pyro_model(x, y, model),
        guide=guide,
        optim=optimizer,
        loss=Trace_ELBO()
    )

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            loss = svi.step(batch_x, batch_y)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f"Época {epoch}/{num_epochs}, Pérdida: {avg_loss:.4f}")

    return losses, guide

# 6️⃣ Predicción con el modelo entrenado usando computación paralela
def _predict_sample(model, x_input, guide):
    """Función auxiliar para predicción paralela."""
    sampled_model = guide()  # Obtener una muestra del modelo posterior
    with torch.no_grad():
        return sampled_model(x_input).cpu().numpy()

def predict(model, x_input, guide, num_samples=100):
    """
    Realiza predicciones con múltiples muestras del modelo posterior.
    
    Args:
        model: Modelo LSTM
        x_input: Datos de entrada
        guide: Guía entrenada
        num_samples: Número de muestras para la predicción
    
    Returns:
        Media y desviación estándar de las predicciones
    """
    num_cores = min(multiprocessing.cpu_count(), num_samples)
    with multiprocessing.Pool(num_cores) as pool:
        predict_fn = partial(_predict_sample, model, x_input, guide)
        predictions = pool.map(predict_fn, range(num_samples))
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

# 7️⃣ Evaluar el modelo
def evaluate_model(model, guide, x_test, y_test, scaler):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Args:
        model: Modelo LSTM
        guide: Guía entrenada
        x_test, y_test: Datos de prueba
        scaler: Scaler para desnormalizar
    
    Returns:
        (rmse, mean_preds_real, std_preds_real, y_test_real)
    """
    mean_preds, std_preds = predict(model, x_test, guide)

    y_test_np = y_test.cpu().numpy()

    # Ajustar forma para inverse_transform
    mean_preds_reshaped = mean_preds.reshape(-1, 1)
    std_preds_reshaped = std_preds.reshape(-1, 1)
    y_test_reshaped = y_test_np.reshape(-1, 1)

    # Desnormalizar
    mean_preds_real = scaler.inverse_transform(mean_preds_reshaped)
    std_preds_real = (
        scaler.inverse_transform(std_preds_reshaped)
        - scaler.inverse_transform(np.zeros_like(std_preds_reshaped))
    )
    y_test_real = scaler.inverse_transform(y_test_reshaped)

    rmse = np.sqrt(np.mean((mean_preds_real - y_test_real) ** 2))
    return rmse, mean_preds_real, std_preds_real, y_test_real

# 8️⃣ Visualizar resultados
def plot_predictions(train_data, test_data, predictions, std_dev, title="Predicciones del modelo"):
    """
    Visualiza las predicciones del modelo junto con intervalos de confianza.
    
    Args:
        train_data: Datos de entrenamiento
        test_data: Datos reales de prueba
        predictions: Predicciones del modelo
        std_dev: Desviación estándar de las predicciones
        title: Título del gráfico
    """
    plt.figure(figsize=(12, 6))
    
    # Datos de entrenamiento
    plt.plot(range(len(train_data)), train_data, label='Datos de entrenamiento')
    
    # Datos de prueba
    test_indices = range(len(train_data), len(train_data) + len(test_data))
    plt.plot(test_indices, test_data, label='Datos reales de prueba')
    
    # Predicciones
    plt.plot(test_indices, predictions, label='Predicciones')
    
    # Intervalo de confianza (95%)
    plt.fill_between(
        test_indices,
        predictions.flatten() - 1.96 * std_dev.flatten(),
        predictions.flatten() + 1.96 * std_dev.flatten(),
        alpha=0.3,
        label='Intervalo de confianza 95%'
    )
    
    plt.title(title)
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 9️⃣ Función principal
def main():
    """Función principal que ejecuta todo el flujo de trabajo."""
    # Configuración
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    seq_length = 30
    hidden_size = 50
    num_layers = 2
    batch_size = 64
    num_epochs = 500
    
    # Obtener y preparar datos
    print(f"Descargando datos para {ticker}...")
    data = get_stock_data(ticker, start_date, end_date)
    
    print("Preparando datos...")
    (x_train, y_train), (x_test, y_test), scaler = prepare_data(data, seq_length)
    
    # Inicializar modelo
    print("Inicializando modelo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianLSTM(
        input_size=1, 
        hidden_size=hidden_size, 
        output_size=1,
        num_layers=num_layers,
        dropout=0.2
    ).to(device)
    
    # Mover datos a dispositivo
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Entrenar modelo
    print(f"Entrenando modelo en {device}...")
    losses, guide = train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Evaluar modelo
    print("Evaluando modelo...")
    rmse, mean_preds, std_preds, y_test_real = evaluate_model(
        model=model,
        guide=guide,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler
    )
    
    print(f"RMSE en conjunto de prueba: {rmse:.2f}")
    
    # Visualizar resultados
    train_data_real = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
    plot_predictions(
        train_data=train_data_real.flatten(),
        test_data=y_test_real.flatten(),
        predictions=mean_preds,
        std_dev=std_preds,
        title=f"Predicción de precios para {ticker}"
    )
    
    # Hacer predicción futura
    print("Realizando predicción futura...")
    last_sequence = x_test[-1].unsqueeze(0)  # Última secuencia de datos
    future_mean, future_std = predict(model, last_sequence, guide)
    future_mean_price = scaler.inverse_transform(future_mean.reshape(-1, 1))
    future_std_price = (
        scaler.inverse_transform(future_std.reshape(-1, 1))
        - scaler.inverse_transform(np.zeros_like(future_std.reshape(-1, 1)))
    )
    print(f"Predicción para el siguiente día: {future_mean_price[0][0]:.2f} ± {future_std_price[0][0]:.2f}")





   


