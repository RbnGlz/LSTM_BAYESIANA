# LSTM_BAYESIANA
Implementación de una red neuronal LSTM bayesiana con PyTorch y Pyro.
Utilizamos los drivers CUDA de nuestra tarjeta gráfica Nvidia.

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
