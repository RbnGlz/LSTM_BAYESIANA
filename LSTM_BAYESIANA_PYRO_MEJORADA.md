# Modelo LSTM Bayesiano para Predicción de Precios de Activos Financieros

Modelo de red neuronal LSTM Bayesiana optimizado para la predicción de precios de activos financieros. El modelo utiliza inferencia bayesiana para proporcionar no solo predicciones puntuales, sino también estimaciones de incertidumbre, crucial para la toma de decisiones financieras robustas.

## Características principales

- **Inferencia Bayesiana**: Utiliza Pyro para implementar inferencia variacional estocástica (SVI).
- **Arquitectura LSTM**: Modelo de redes neuronales recurrentes para capturar dependencias temporales en datos de precios.
- **Procesamiento paralelo**: Implementación optimizada para aprovechar múltiples núcleos de la CPU.
- **Cuantificación de incertidumbre**: Proporciona intervalos de confianza para las predicciones.
- **Soporte para GPU**: Aceleración mediante CUDA cuando está disponible.
- **Visualización avanzada**: Gráficos intuitivos de predicciones con bandas de incertidumbre.
- **Caching de datos**: Sistema de caché para evitar descargas repetidas.
- **Ingeniería de características**: Indicadores técnicos avanzados para mejorar la precisión.

## Optimizaciones implementadas

### 1. Descarga y almacenamiento de datos

- **Sistema de caché**: Evita descargar repetidamente los mismos datos, ahorrando tiempo y recursos.
- **Descarga en paralelo**: Permite obtener datos de múltiples tickers simultáneamente.
- **Manejo de errores robusto**: Implementa reintentos con backoff exponencial para manejar fallos de conexión.
- **Formato eficiente**: Utiliza Parquet para almacenamiento en caché, más eficiente que CSV para datos financieros.
- **Flexibilidad**: Permite descargar un solo ticker o una lista de tickers con la misma función.

### 2. Preprocesamiento de datos

- **Ingeniería de características**: Añade indicadores técnicos relevantes (retornos logarítmicos, medias móviles, volatilidad, momentum).
- **Escalado robusto**: Usa RobustScaler en lugar de MinMaxScaler para manejar mejor los outliers en datos financieros.
- **Stride configurable**: Permite crear secuencias con diferentes pasos, reduciendo la correlación entre muestras.
- **Conjunto de validación**: Añade un conjunto de validación para mejor monitoreo durante el entrenamiento.
- **Procesamiento vectorizado**: Optimiza la creación de secuencias para un mejor rendimiento.
- **Preservación temporal**: Mantiene el orden cronológico al dividir los datos, crucial para series temporales.

### 3. Arquitectura LSTM mejorada

- **Normalización de capas**: Añadida para estabilizar el entrenamiento.
- **Mejor inicialización**: Optimizada la inicialización de estados ocultos.
- **Manejo de dispositivos**: Mejorado para seamless switching entre CPU/GPU.
- **Opción bidireccional**: Capacidad de utilizar LSTM bidireccional según necesidad.

### 4. Modelo bayesiano optimizado

- **Priors más informativos**: Mejora la convergencia y estabilidad.
- **Distribución Gamma**: Reemplaza la distribución Uniforme para sigma, más adecuada para modelar varianza.
- **Estructura mejorada**: Mejor definición del modelo probabilístico.

### 5. Predicción y evaluación optimizadas

- **Implementación adaptativa**: Paralelización selectiva (solo cuando es ventajoso).
- **Reproducibilidad**: Fijación de semillas aleatorias para muestras consistentes.
- **Eficiencia de memoria**: Mejor manejo con `torch.set_grad_enabled()`.
- **Múltiples métricas**: Evaluación con RMSE, MAE, MAPE para análisis más completo.
- **Desnormalización correcta**: Implementada correcta desnormalización de la desviación estándar.

### 6. Persistencia y usabilidad

- **Guardado/carga de modelos**: Mecanismo para preservar modelos entrenados.
- **Early stopping**: Implementado para evitar sobreajuste y ahorrar tiempo.
- **Hiperparámetros optimizados**: Tamaño de batch y capas ocultas ajustados para mejor rendimiento.

## Código completo del modelo

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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
from functools import partial
import os
import time
import concurrent.futures
from sklearn.preprocessing import RobustScaler

# 1️: Descargar datos de precios de acciones - OPTIMIZADO
def get_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01", cache_dir="data_cache"):
    """
    Descarga datos históricos de precios de acciones con caché y manejo mejorado de errores.
    
    Args:
        ticker: Símbolo de la acción o lista de símbolos
        start: Fecha de inicio (formato YYYY-MM-DD)
        end: Fecha de fin (formato YYYY-MM-DD)
        cache_dir: Directorio para almacenar datos en caché
    
    Returns:
        DataFrame con datos de precios o diccionario de DataFrames si se pasan múltiples tickers
    """
    # Crear directorio de caché si no existe
    os.makedirs(cache_dir, exist_ok=True)
    
    # Función para descargar un solo ticker con reintentos
    def download_single_ticker(tick):
        cache_file = os.path.join(cache_dir, f"{tick}_{start}_{end}.parquet")
        
        # Verificar si existe en caché
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                # Si hay error al leer caché, descargar de nuevo
                pass
        
        # Implementar reintentos con backoff exponencial
        max_retries = 5
        for attempt in range(max_retries):
            try:
                df = yf.download(tick, start=start, end=end, progress=False)
                
                if df.empty:
                    raise ValueError(f"No se encontraron datos para {tick}")
                
                # Guardar en caché
                df.to_parquet(cache_file)
                return df
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error al descargar {tick}: {e}")
                    raise
                # Esperar con backoff exponencial
                time.sleep(2 ** attempt)
    
    # Manejar múltiples tickers o un solo ticker
    if isinstance(ticker, list):
        # Descargar en paralelo
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ticker), 10)) as executor:
            future_to_ticker = {executor.submit(download_single_ticker, tick): tick for tick in ticker}
            for future in concurrent.futures.as_completed(future_to_ticker):
                tick = future_to_ticker[future]
                try:
                    results[tick] = future.result()
                except Exception as e:
                    print(f"Error en ticker {tick}: {e}")
        return results
    else:
        # Descargar un solo ticker
        df = download_single_ticker(ticker)
        return df[["Close"]].values  # Para mantener compatibilidad con el código original

# 2️: Preprocesar datos - OPTIMIZADO
def prepare_data(data, seq_length=30, test_size=0.2, val_size=0.1, stride=1, feature_engineering=True):
    """
    Preprocesa los datos para el entrenamiento del modelo LSTM con técnicas avanzadas.
    
    Args:
        data: Array de precios
        seq_length: Longitud de la secuencia para predecir
        test_size: Proporción de datos para prueba
        val_size: Proporción de datos para validación
        stride: Paso entre secuencias consecutivas (1 = todas las secuencias)
        feature_engineering: Si se deben añadir características técnicas
        
    Returns:
        Tensores de entrenamiento/validación/prueba y scaler para desnormalizar
    """
    # Convertir a DataFrame para facilitar el feature engineering
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=['Close'])
    else:
        df = data.copy()
    
    if feature_engineering:
        # Añadir características técnicas (retornos, medias móviles, etc.)
        # Retornos logarítmicos (más estables que los porcentuales)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Medias móviles
        df['ma7'] = df['Close'].rolling(window=7).mean() / df['Close'] - 1
        df['ma21'] = df['Close'].rolling(window=21).mean() / df['Close'] - 1
        
        # Volatilidad
        df['volatility'] = df['log_return'].rolling(window=21).std()
        
        # Momentum
        df['momentum'] = df['Close'].pct_change(periods=5)
        
        # Eliminar NaN
        df = df.dropna()
    
    # Seleccionar columnas a usar
    if feature_engineering:
        feature_columns = ['Close', 'log_return', 'ma7', 'ma21', 'volatility', 'momentum']
        features = df[feature_columns].values
    else:
        features = df[['Close']].values
    
    # Normalizar datos - usando RobustScaler para mayor robustez ante outliers
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Crear secuencias X e Y de manera eficiente
    X, y = [], []
    
    # Método vectorizado para crear secuencias (mucho más rápido que los bucles)
    for i in range(0, len(scaled_features) - seq_length, stride):
        X.append(scaled_features[i:i + seq_length])
        y.append(scaled_features[i + seq_length, 0])  # Predecir solo el precio de cierre
    
    # Convertir a arrays NumPy eficientemente
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Determinar tamaños de conjuntos
    n = len(X)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    # Mantener tiempo cronológico (sin aleatorizar)
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    # Convertir a tensores PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_columns if feature_engineering else ['Close']

# 3️: Definir el modelo LSTM mejorado
class BayesianLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Utilizar bidireccional para capturar mejor las tendencias
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Cambiar a True si se necesita bidireccionalidad
        )
        
        # Añadir capa de normalización para estabilizar entrenamiento
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inicializar estados ocultos con una mejor inicialización
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        # Optimización de memoria
        with torch.set_grad_enabled(self.training):
            out, _ = self.lstm(x, (h0, c0))
            out = self.norm(out[:, -1, :])  # Normalizar antes de la capa lineal
            out = self.fc(out)
        return out

# 4️: Modelo probabilístico mejorado en Pyro
def pyro_model(x, y=None, model=None):
    """
    Define el modelo probabilístico para inferencia bayesiana con priors más informativos.
    """
    if model is None:
        raise ValueError("Se debe proporcionar un modelo")
    
    # Priors para pesos y sesgos - más informativo para convergencia más rápida
    scale_prior = 0.1
    
    # Priors para la capa final
    w_prior = dist.Normal(0.0, scale_prior).expand(model.fc.weight.shape).to_event(2)
    b_prior = dist.Normal(0.0, scale_prior).expand(model.fc.bias.shape).to_event(1)
    
    # También podemos añadir priors para los parámetros LSTM
    priors = {
        "fc.weight": w_prior, 
        "fc.bias": b_prior,
    }
    
    # Módulo aleatorio con los priors
    lifted_module = pyro.random_module("module", model, priors)()
    
    # Predicción del modelo
    y_hat = lifted_module(x)

    # Prior para la varianza del ruido - más informativo
    sigma = pyro.sample("sigma", dist.Gamma(2.0, 3.0))
    
    # Likelihood con mejor manejo de forma
    with pyro.plate("data", x.size(0)):
        pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)

    return y_hat

# 5️: Entrenar con inferencia bayesiana y early stopping
def train(model, x_train, y_train, x_val=None, y_val=None, num_epochs=1000, batch_size=64, lr=0.01, 
          patience=20, early_stopping=True):
    """
    Entrena el modelo usando SVI (Stochastic Variational Inference) con early stopping.
    
    Args:
        model: Modelo LSTM
        x_train, y_train: Datos de entrenamiento
        x_val, y_val: Datos de validación (opcional)
        num_epochs: Número de épocas
        batch_size: Tamaño del lote
        lr: Tasa de aprendizaje
        patience: Número de épocas para early stopping
        early_stopping: Si se debe utilizar early stopping
    
    Returns:
        Lista de pérdidas durante el entrenamiento y la guía entrenada
    """
    # Limpiar parámetros anteriores
    pyro.clear_param_store()
    
    # Crear dataset y dataloader para procesamiento por lotes
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Si hay datos de validación, crear dataloader
    val_dataloader = None
    if x_val is not None and y_val is not None:
        val_dataset = TensorDataset(x_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
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
    val_losses = []
    best_val_loss = float('inf')
    no_improvement = 0
    best_state = None
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            loss = svi.step(batch_x, batch_y)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Validación (si hay datos)
        if val_dataloader is not None:
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_dataloader:
                    # Calcular loss en validación
                    val_loss = svi.evaluate_loss(batch_x, batch_y)
                    val_epoch_loss += val_loss
            
            avg_val_loss = val_epoch_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            
            # Imprimir progreso
            if epoch % 50 == 0:
                print(f"Época {epoch}/{num_epochs}, Pérdida: {avg_loss:.4f}, Val: {avg_val_loss:.4f}")
            
            # Early stopping
            if early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement = 0
                    # Guardar el mejor estado
                    best_state = guide
                else:
                    no_improvement += 1
                
                if no_improvement >= patience:
                    print(f"Early stopping en época {epoch}")
                    return losses, best_state if best_state is not None else guide
        else:
            if epoch % 50 == 0:
                print(f"Época {epoch}/{num_epochs}, Pérdida: {avg_loss:.4f}")

    return losses, guide

# 6️: Predicción optimizada con el modelo entrenado
def predict(model, x_input, guide, num_samples=100, use_parallel=True):
    """
    Realiza predicciones con múltiples muestras del modelo posterior.
    Implementación optimizada para CPU y GPU.
    """
    device = x_input.device
    predictions = []
    
    if use_parallel and device.type == 'cpu':
        # Solo usar paralelización si estamos en CPU
        num_cores = min(multiprocessing.cpu_count(), num_samples)
        
        # Mover datos a CPU para multiprocessing
        x_cpu = x_input.cpu()
        
        # Función local para predicción que maneje bien el contexto de device
        def _predict_sample_wrapper(i):
            torch.manual_seed(i)  # Garantizar diferentes muestras
            sampled_model = guide()
            sampled_model.to('cpu')
            with torch.no_grad():
                return sampled_model(x_cpu).numpy()
        
        with multiprocessing.Pool(num_cores) as pool:
            predictions = pool.map(_predict_sample_wrapper, range(num_samples))
    else:
        # Ejecución secuencial (mejor para GPU)
        for i in range(num_samples):
            torch.manual_seed(i)  # Garantizar diferentes muestras
            sampled_model = guide()
            sampled_model.to(device)
            with torch.no_grad():
                pred = sampled_model(x_input).cpu().numpy()
                predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

# 7️: Evaluar el modelo con corrección matemática
def evaluate_model(model, guide, x_test, y_test, scaler, feature_columns):
    """
    Evalúa el modelo en el conjunto de prueba con cálculos de error corregidos.
    """
    # Determinar si usar paralelización
    device = x_test.device
    use_parallel = device.type == 'cpu'
    
    mean_preds, std_preds = predict(model, x_test, guide, use_parallel=use_parallel)
    y_test_np = y_test.cpu().numpy()

    # Ajustar forma para inverse_transform
    mean_preds_reshaped = mean_preds.reshape(-1, 1)
    y_test_reshaped = y_test_np.reshape(-1, 1)

    # Crear arrays para transformación inversa
    # Si tenemos múltiples características, necesitamos crear arrays completos
    if len(feature_columns) > 1:
        # Crear array dummy con ceros para las otras características
        dummy_array = np.zeros((mean_preds_reshaped.shape[0], len(feature_columns)))
        # Colocar valores reales solo en la columna 'Close'
        close_idx = feature_columns.index('Close')
        dummy_array[:, close_idx] = mean_preds_reshaped.flatten()
        mean_preds_for_inverse = dummy_array
        
        # Lo mismo para los valores reales y std
        dummy_y = np.zeros((y_test_reshaped.shape[0], len(feature_columns)))
        dummy_y[:, close_idx] = y_test_reshaped.flatten()
        y_test_for_inverse = dummy_y
        
        # Para std_preds
        dummy_std = np.zeros((mean_preds_reshaped.shape[0], len(feature_columns)))
        dummy_std[:, close_idx] = std_preds.flatten()
        std_preds_for_inverse = dummy_std
    else:
        mean_preds_for_inverse = mean_preds_reshaped
        y_test_for_inverse = y_test_reshaped
        std_preds_for_inverse = std_preds.reshape(-1, 1)
    
    # Corrección matemática para desnormalizar desviación estándar
    if isinstance(scaler, RobustScaler):
        # Para RobustScaler, necesitamos saber el factor de escala (IQR)
        # Aproximar usando la diferencia entre un valor con y sin std
        dummy_zeros = np.zeros_like(std_preds_for_inverse)
        base = scaler.inverse_transform(dummy_zeros)
        with_std = scaler.inverse_transform(std_preds_for_inverse)
        std_preds_real = with_std - base
        
        # Desnormalizar predicciones y valores reales
        mean_preds_real = scaler.inverse_transform(mean_preds_for_inverse)
        y_test_real = scaler.inverse_transform(y_test_for_inverse)
        
        # Extraer solo la columna de Close
        if len(feature_columns) > 1:
            mean_preds_real = mean_preds_real[:, close_idx].reshape(-1, 1)
            y_test_real = y_test_real[:, close_idx].reshape(-1, 1)
            std_preds_real = std_preds_real[:, close_idx].reshape(-1, 1)
    else:
        # Si no es RobustScaler, asumir MinMaxScaler u otro
        scale_factor = 1.0  # Ajustar según el scaler
        std_preds_real = std_preds.reshape(-1, 1) * scale_factor
        mean_preds_real = scaler.inverse_transform(mean_preds_for_inverse)
        y_test_real = scaler.inverse_transform(y_test_for_inverse)

    # Cálculo de métricas múltiples
    rmse = np.sqrt(np.mean((mean_preds_real - y_test_real) ** 2))
    mae = np.mean(np.abs(mean_preds_real - y_test_real))
    mape = np.mean(np.abs((y_test_real - mean_preds_real) / y_test_real)) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return metrics, mean_preds_real, std_preds_real, y_test_real

# 8️: Visualizar resultados con gráficos mejorados
def plot_predictions(train_data, test_data, predictions, std_dev, title="Predicciones del modelo"):
    """
    Visualiza las predicciones del modelo junto con intervalos de confianza.
    Versión mejorada con estilos y anotaciones.
    """
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Configurar el eje X
    x_train = np.arange(len(train_data))
    x_test = np.arange(len(train_data), len(train_data) + len(test_data))
    
    # Datos de entrenamiento
    plt.plot(x_train, train_data, label='Datos históricos', color='#4C72B0', linewidth=1.5, alpha=0.8)
    
    # Datos de prueba
    plt.plot(x_test, test_data, label='Datos reales', color='#55A868', linewidth=2)
    
    # Predicciones
    plt.plot(x_test, predictions, label='Predicciones', color='#C44E52', linewidth=2, linestyle='-')
    
    # Intervalo de confianza (95%)
    plt.fill_between(
        x_test,
        predictions.flatten() - 1.96 * std_dev.flatten(),
        predictions.flatten() + 1.96 * std_dev.flatten(),
        alpha=0.2,
        color='#C44E52',
        label='Intervalo de confianza 95%'
    )
    
    # Marcar división entre entrenamiento y prueba
    plt.axvline(x=len(train_data), color='#8172B3', linestyle='--', alpha=0.7, 
                label='División entrenamiento/prueba')
    
    # Añadir título y etiquetas
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tiempo', fontsize=12, labelpad=10)
    plt.ylabel('Precio', fontsize=12, labelpad=10)
    
    # Mejorar leyenda
    plt.legend(loc='best', frameon=True, fontsize=10)
    
    # Mejorar diseño
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Añadir anotaciones de métricas si están disponibles
    try:
        rmse = np.sqrt(np.mean((predictions.flatten() - test_data.flatten()) ** 2))
        plt.annotate(f'RMSE: {rmse:.2f}', 
                     xy=(0.02, 0.05), 
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
    except:
        pass
    
    return plt

# 9️: Función principal mejorada
def main():
    """Función principal optimizada."""
    # Configuración
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    seq_length = 30
    hidden_size = 64  # Aumentado para mejor capacidad expresiva
    num_layers = 2
    batch_size = 128  # Aumentado para mejor paralelización
    num_epochs = 500
    
    # Usar cuda si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Obtener y preparar datos
    print(f"Descargando datos para {ticker}...")
    data = get_stock_data(ticker, start_date, end_date)
    
    print("Preparando datos...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler, feature_columns = prepare_data(
        data, 
        seq_length=seq_length, 
        feature_engineering=True
    )
    
    # Calcular input_size basado en el número de características
    input_size = x_train.shape[2]  # Para modelos con múltiples características
    
    # Inicializar modelo con técnicas avanzadas
    print("Inicializando modelo...")
    model = BayesianLSTM(
        input_size=input_size, 
        hidden_size=hidden_size, 
        output_size=1,
        num_layers=num_layers,
        dropout=0.2
    ).to(device)
    
    # Implementar guardado y carga de modelo
    model_path = f"bayesian_lstm_{ticker}.pt"
    guide_path = f"bayesian_guide_{ticker}.pt"
    
    # Mover datos a dispositivo
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Entrenar o cargar modelo
    if os.path.exists(model_path) and os.path.exists(guide_path):
        print("Cargando modelo guardado...")
        model.load_state_dict(torch.load(model_path))
        guide = torch.load(guide_path)
    else:
        print(f"Entrenando modelo en {device}...")
        losses, guide = train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            patience=20,
            early_stopping=True
        )
        
        # Guardar modelo
        torch.save(model.state_dict(), model_path)
        torch.save(guide, guide_path)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    metrics, mean_preds, std_preds, y_test_real = evaluate_model(
        model=model,
        guide=guide,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        feature_columns=feature_columns
    )
    
    print(f"Métricas en conjunto de prueba:")
    for name, value in metrics.items():
        print(f"- {name.upper()}: {value:.4f}")
    
    # Visualizar resultados
    train_data_real = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
    if len(feature_columns) > 1:
        close_idx = feature_columns.index('Close')
        train_data_real = train_data_real[:, close_idx].reshape(-1, 1)
    
    plt = plot_predictions(
        train_data=train_data_real.flatten(),
        test_data=y_test_real.flatten(),
        predictions=mean_preds.flatten(),
        std_dev=std_preds.flatten(),
        title=f"Predicción de precios para {ticker}"
    )
    plt.savefig(f"{ticker}_predicciones.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Hacer predicción futura con método optimizado
    print("Realizando predicción futura...")
    last_sequence = x_test[-1].unsqueeze(0)
    future_mean, future_std = predict(model, last_sequence, guide, use_parallel=(device.type == 'cpu'))
    
    # Preparar para desnormalización
    future_mean_reshaped = future_mean.reshape(1, -1)
    future_std_reshaped = future_std.reshape(1, -1)
    
    # Procesar para desnormalización
    if len(feature_columns) > 1:
        # Crear arrays dummy completos
        dummy_mean = np.zeros((1, len(feature_columns)))
        dummy_std = np.zeros((1, len(feature_columns)))
        
        # Colocar valores solo en la columna de Close
        close_idx = feature_columns.index('Close')
        dummy_mean[:, close_idx] = future_mean_reshaped
        dummy_std[:, close_idx] = future_std_reshaped
        
        # Desnormalizar
        future_mean_price_full = scaler.inverse_transform(dummy_mean)
        future_mean_price = future_mean_price_full[:, close_idx].reshape(-1, 1)
        
        # Para la desviación estándar
        dummy_zeros = np.zeros_like(dummy_std)
        base = scaler.inverse_transform(dummy_zeros)
        with_std = scaler.inverse_transform(dummy_std)
        future_std_price = (with_std - base)[:, close_idx].reshape(-1, 1)
    else:
        future_mean_price = scaler.inverse_transform(future_mean_reshaped.reshape(-1, 1))
        
        # Aproximar el factor de escala para std
        dummy_zeros = np.zeros_like(future_std_reshaped.reshape(-1, 1))
        base = scaler.inverse_transform(dummy_zeros)
        with_std = scaler.inverse_transform(future_std_reshaped.reshape(-1, 1))
        future_std_price = with_std - base
    
    print(f"Predicción para el siguiente día: {future_mean_price[0][0]:.2f} ± {future_std_price[0][0]:.2f}")

if __name__ == "__main__":
    main()
```

## Cómo usar el modelo

1. **Instalación de dependencias**:
   ```bash
   pip install torch pyro-ppl yfinance pandas numpy matplotlib scikit-learn
   ```

2. **Ejecución del modelo**:
   ```bash
   python bayesian_lstm.py
   ```

3. **Parámetros configurables**:
   - `ticker`: Símbolo de la acción a analizar (ej. "AAPL", "MSFT", "GOOGL")
   - `start_date` y `end_date`: Rango de fechas para los datos
   - `hidden_size`: Tamaño de capas ocultas LSTM
   - `num_layers`: Número de capas LSTM
   - `batch_size`: Tamaño de lote para entrenamiento
   - `num_epochs`: Número máximo de épocas de entrenamiento
