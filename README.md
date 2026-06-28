# Proyecto 2 — Herramientas de predicción ML para el agente eléctrico

**IE0435 Inteligencia Artificial Aplicada a la Ingeniería Eléctrica · I-2026**

Este proyecto **extiende** el agente de predicción de series de tiempo de
[ErickVM7/Proyecto-Electrico](https://github.com/ErickVM7/Proyecto-Electrico),
agregando dos herramientas adicionales solicitadas en el enunciado del
Proyecto 2:

| Herramienta nueva | Técnica | Archivo |
|---|---|---|
| `predict_knn`  | K-Nearest Neighbors (regresión) | `herramientas_ml.py` |
| `predict_tree` | Árbol de regresión (Decision Tree) | `herramientas_ml.py` |

El agente original usaba solo modelos estadísticos (Prophet y ARIMA/SARIMA).
Estas herramientas incorporan **modelos de aprendizaje automático clásico**,
integrados al mismo flujo de herramientas del agente LLM local (Ollama).

## Archivos de este aporte

```
Proyecto2/
├── herramientas_ml.py   # predict_knn y predict_tree (+ esquemas de tool)
├── main3.py             # agente integrado (extiende main2.py de Erick)
├── evaluacion.py        # backtesting: KNN vs Árbol vs baseline ICE (MW_P)
├── datos_limpios.csv    # dataset CENCE (demanda nacional, 15 min)
├── requirements.txt
└── README.md
```

## ¿Cómo predicen KNN y los árboles una serie de tiempo?

KNN y los árboles **no** son modelos de series de tiempo. Para que puedan
predecir demanda se construye una matriz de características (*feature
engineering*):

- **Lags**: valores rezagados de la propia serie (`lag_1..lag_4` = última hora,
  `lag_96` = mismo instante del día anterior). Dan memoria de corto plazo y
  estacionalidad diaria.
- **Calendario**: hora, minuto, día de la semana y fin de semana. Capturan el
  patrón de consumo según la hora y el día.

La predicción a varios pasos es **recursiva**: cada valor predicho se reinyecta
como lag para predecir el siguiente intervalo de 15 minutos.

## Uso

### 1. Vía el agente (lenguaje natural)

```bash
python main3.py
```

```
🙂 > predice MW para los próximos 2 días con KNN
🙂 > predice MW de mañana usando un árbol de regresión
```

### 2. Vía código (sin LLM)

```python
import pandas as pd
from herramientas_ml import predict_knn, predict_tree

dtf = pd.read_csv("datos_limpios.csv")
dtf["fechaHora"] = pd.to_datetime(dtf["fechaHora"])
dtf = dtf.set_index("fechaHora").sort_index().asfreq("15min")

predict_knn(dtf, column="MW", horizon=2)        # 2 días
predict_tree(dtf, column="MW", horizon=1)       # 1 día
```

### 3. Evaluación / backtesting

```bash
python evaluacion.py
```

Entrena con todo el histórico menos el último día, predice ese día y compara
contra los valores reales y contra la demanda programada del ICE (`MW_P`).

## Resultados (backtesting, 1 día = 96 pasos, columna MW)

| Modelo | MAE | RMSE | MAPE % |
|---|---|---|---|
| **Árbol de regresión** | **26.33** | **36.13** | **1.60** |
| KNN (k=15) | 36.48 | 39.00 | 2.45 |
| ICE (MW_P, baseline) | 47.26 | 59.06 | 3.05 |

En esta ventana ambos modelos superan a la demanda programada del propio ICE.
*(Los valores pueden variar según la fecha de corte y los datos disponibles.)*

## Requisitos

```bash
pip install -r requirements.txt
```

Para el chat con el agente se requiere además [Ollama](https://ollama.com/download)
con un modelo local (p. ej. `qwen2.5:7b`). Las herramientas de predicción y el
backtesting funcionan sin Ollama.
