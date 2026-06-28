import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Pasos por día con frecuencia de 15 min (24 h * 4 = 96).
PASOS_POR_DIA = 96

# Lags por defecto: cubren la última hora (4 pasos) + el valor de hace un día
# completo (96 pasos), lo que le da al modelo memoria diaria.
LAGS_DEFAULT = [1, 2, 3, 4, PASOS_POR_DIA]


# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------
def _interpretar_horizonte(horizon, end_date, last_date):
    """
    Convierte el horizonte solicitado (en días o como fecha) a un número de
    pasos de 15 minutos.
    """
    steps = PASOS_POR_DIA  # por defecto, 1 día
    if horizon is not None:
        if isinstance(horizon, str):
            nums = re.findall(r"\d+", horizon)
            dias = int(nums[0]) if nums else 1
            steps = max(1, dias * PASOS_POR_DIA)
        elif isinstance(horizon, int):
            steps = max(1, horizon * PASOS_POR_DIA)

    if end_date:
        try:
            objetivo = pd.to_datetime(end_date)
            delta = objetivo - last_date
            if delta.total_seconds() > 0:
                steps = max(1, int(delta.total_seconds() / (15 * 60)))
        except Exception:
            pass
    return steps


def _features_calendario(indice):
    """
    Construye las variables de calendario a partir de un índice datetime.
    """
    return pd.DataFrame(
        {
            "hora": indice.hour,
            "minuto": indice.minute,
            "dia_semana": indice.dayofweek,
            "fin_de_semana": (indice.dayofweek >= 5).astype(int),
        },
        index=indice,
    )


def _construir_matriz(serie, lags):
    """
    Arma la matriz de entrenamiento (X, y) a partir de una serie temporal:
    columnas de lags + columnas de calendario.
    """
    df = pd.DataFrame({"y": serie})
    for lag in lags:
        df[f"lag_{lag}"] = serie.shift(lag)

    cal = _features_calendario(df.index)
    df = pd.concat([df, cal], axis=1).dropna()

    columnas_x = [f"lag_{lag}" for lag in lags] + list(cal.columns)
    X = df[columnas_x]
    y = df["y"]
    return X, y, columnas_x


def _predecir_recursivo(modelo, serie, lags, columnas_x, steps, freq="15min"):
    """
    Predicción multi-paso recursiva
    """
    historia = list(serie.values)
    ultima_fecha = serie.index[-1]
    fechas_futuras = pd.date_range(ultima_fecha, periods=steps + 1, freq=freq)[1:]
    predicciones = []

    for fecha in fechas_futuras:
        fila = {}
        for lag in lags:
            fila[f"lag_{lag}"] = historia[-lag]
        fila["hora"] = fecha.hour
        fila["minuto"] = fecha.minute
        fila["dia_semana"] = fecha.dayofweek
        fila["fin_de_semana"] = 1 if fecha.dayofweek >= 5 else 0

        X_fila = pd.DataFrame([fila])[columnas_x]
        y_pred = float(modelo.predict(X_fila)[0])
        predicciones.append(y_pred)
        historia.append(y_pred)

    return pd.DataFrame({"fecha": fechas_futuras, "MW_pred": predicciones})


def _graficar_y_guardar(out_df, column, nombre_modelo, steps, save_dir):
    """Grafica las predicciones y las guarda en CSV."""
    plt.figure(figsize=(10, 5))
    plt.plot(out_df["fecha"], out_df["MW_pred"], "o-",
             label=f"Predicción ({nombre_modelo})")
    plt.title(f"Predicción {nombre_modelo} para {column} "
              f"({steps} pasos, {steps // PASOS_POR_DIA} día(s))")
    plt.xlabel("Fecha")
    plt.ylabel(column)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs(save_dir, exist_ok=True)
    inicio = pd.to_datetime(out_df["fecha"].min()).strftime("%Y%m%d_%H%M")
    fin = pd.to_datetime(out_df["fecha"].max()).strftime("%Y%m%d_%H%M")
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"pred_{nombre_modelo.lower()}_{column}_{inicio}_to_{fin}_{ts}.csv"
    ruta = os.path.join(save_dir, nombre)
    out_df[["fecha", "MW_pred"]].to_csv(ruta, index=False)
    return ruta


# -----------------------------------------------------------------------------
# Herramienta 1: KNN (regresión)
# -----------------------------------------------------------------------------
def predict_knn(dtf, column=None, horizon=None, end_date=None,
                n_neighbors=15, lags=None, save_dir="predicciones_ml"):
    """
    Predice valores futuros de una serie con K-Nearest Neighbors.

    KNN predice promediando los `n_neighbors` casos históricos más parecidos
    en el espacio de características (lags + calendario). Se usa un pipeline con
    StandardScaler porque KNN depende de distancias y las escalas de los lags
    (cientos de MW) y del calendario (0-23, 0-6) son muy distintas.
    """
    try:
        if column is None or column not in dtf.columns:
            return (f"Error: especifica una columna válida. "
                    f"Columnas disponibles: {list(dtf.columns)}")

        lags = lags or LAGS_DEFAULT
        serie = dtf[column].dropna()
        last_date = serie.index[-1]
        steps = _interpretar_horizonte(horizon, end_date, last_date)

        X, y, columnas_x = _construir_matriz(serie, lags)
        modelo = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"),
        )
        modelo.fit(X, y)

        out_df = _predecir_recursivo(modelo, serie, lags, columnas_x, steps)
        ruta = _graficar_y_guardar(out_df, column, "KNN", steps, save_dir)

        print("\n📈 Predicciones futuras (KNN):\n")
        print(out_df.to_string(index=False))
        print(f"\n💾 Guardado en: {ruta}")
        return (f"Predicción KNN completada ({len(out_df)} puntos, "
                f"k={n_neighbors}). Archivo: {ruta}")
    except Exception as e:
        return f"Error durante la predicción KNN: {e}"