# =============================================================================
# evaluacion.py
# -----------------------------------------------------------------------------
# Backtesting de las herramientas de predicción ML (KNN y árbol de regresión)
# frente a la demanda real y frente a la demanda programada del ICE (columna
# MW_P, que actúa como baseline "oficial").
#
# Metodología:
#   - Se reserva el último tramo de la serie como conjunto de prueba (test).
#   - Se entrena cada modelo SOLO con los datos anteriores (train).
#   - Se predice el tramo de prueba de forma recursiva y se comparan las
#     predicciones contra los valores reales con MAE, RMSE y MAPE.
#
# =============================================================================
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
from herramientas_ml import (
    _construir_matriz,
    _predecir_recursivo,
    LAGS_DEFAULT,
    PASOS_POR_DIA,
)
 
DATA_PATH = "datos_limpios.csv"
 
# Fechas de corte distribuidas a lo largo del año disponible.
# El modelo se entrena con todo lo anterior y se prueba en el día siguiente.
# No hay que ir al ICE: son fechas dentro del CSV que ya tenemos.
FECHAS_CORTE = [
    "2024-11-15",   # fin de año (temporada seca inicio)
    "2025-01-20",   # enero
    "2025-03-10",   # marzo (semana santa cerca)
    "2025-05-05",   # inicio lluviosa
    "2025-07-01",   # julio
    "2025-08-25",   # fin del histórico
]
 
 
def cargar_datos(path=DATA_PATH):
    dtf = pd.read_csv(path)
    dtf["fechaHora"] = pd.to_datetime(dtf["fechaHora"], errors="coerce")
    dtf.set_index("fechaHora", inplace=True)
    dtf.sort_index(inplace=True)
    dtf = dtf.asfreq("15min")
    return dtf
 
 
def metricas(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae  = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = np.mean(np.abs((real - pred) / real)) * 100
    return mae, rmse, mape
 
 
def backtest_multifecha(dtf, column="MW", fechas=FECHAS_CORTE,
                        n_neighbors=15, max_depth=12):
    """
    Corre el backtesting en varias fechas de corte y devuelve:
      - df_detalle : métricas por fecha y modelo
      - df_promedio: métricas promediadas sobre todas las fechas
    """
    serie = dtf[column].dropna()
    resultados = []
 
    for fecha in fechas:
        idx = serie.index.searchsorted(pd.Timestamp(fecha))
        train = serie.iloc[:idx]
        test  = serie.iloc[idx : idx + PASOS_POR_DIA]
 
        if len(train) < 500 or len(test) < PASOS_POR_DIA:
            print(f" {fecha}: datos insuficientes, se omite.")
            continue
 
        X, y, cx = _construir_matriz(train, LAGS_DEFAULT)
 
        # KNN
        m_knn = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"),
        ).fit(X, y)
        pred_knn = _predecir_recursivo(
            m_knn, train, LAGS_DEFAULT, cx, PASOS_POR_DIA
        )["MW_pred"].values
 
        # Árbol de regresión
        m_tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=5, random_state=0
        ).fit(X, y)
        pred_tree = _predecir_recursivo(
            m_tree, train, LAGS_DEFAULT, cx, PASOS_POR_DIA
        )["MW_pred"].values
 
        # Baseline ICE
        pred_ice = dtf["MW_P"].iloc[idx : idx + PASOS_POR_DIA].values
 
        for nombre, pred in [
            ("KNN", pred_knn),
            ("Arbol", pred_tree),
            ("ICE (baseline)", pred_ice),
        ]:
            mae, rmse, mape = metricas(test.values, pred)
            resultados.append({
                "Fecha": fecha, "Modelo": nombre,
                "MAE": mae, "RMSE": rmse, "MAPE_%": mape,
            })
 
    df_detalle  = pd.DataFrame(resultados)
    df_promedio = (
        df_detalle.groupby("Modelo")[["MAE", "RMSE", "MAPE_%"]]
        .mean()
        .round(2)
        .sort_values("MAPE_%")
    )
    return df_detalle, df_promedio
 
 
def graficar_multifecha(dtf, column="MW", fechas=FECHAS_CORTE,
                        n_neighbors=15, max_depth=12,
                        guardar="backtest_multifecha.png"):
    """
    Grilla de subplots: un panel por fecha de corte, mostrando real vs modelos.
    """
    serie = dtf[column].dropna()
    n = len(fechas)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=False)
    axes = axes.flatten()
 
    for i, fecha in enumerate(fechas):
        ax = axes[i]
        idx = serie.index.searchsorted(pd.Timestamp(fecha))
        test  = serie.iloc[idx : idx + PASOS_POR_DIA]
        train = serie.iloc[:idx]
        if len(train) < 500 or len(test) < PASOS_POR_DIA:
            ax.set_visible(False); continue
 
        X, y, cx = _construir_matriz(train, LAGS_DEFAULT)
        m_knn = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"),
        ).fit(X, y)
        m_tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=5, random_state=0
        ).fit(X, y)
        pk = _predecir_recursivo(m_knn,  train, LAGS_DEFAULT, cx, PASOS_POR_DIA)["MW_pred"].values
        pt = _predecir_recursivo(m_tree, train, LAGS_DEFAULT, cx, PASOS_POR_DIA)["MW_pred"].values
        pi = dtf["MW_P"].iloc[idx : idx + PASOS_POR_DIA].values
 
        horas = range(len(test))
        ax.plot(horas, test.values, "k-",  lw=2,  label="Real")
        ax.plot(horas, pk,          "--",  label="KNN")
        ax.plot(horas, pt,          "--",  label="Árbol")
        ax.plot(horas, pi,          ":",   label="ICE")
        ax.set_title(fecha, fontsize=10)
        ax.set_xlabel("Intervalo (15 min)")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
 
    fig.suptitle(f"Backtesting — {column} — 6 fechas de corte", fontsize=13)
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150)
    plt.show()
    return guardar
 
 
if __name__ == "__main__":
    dtf = cargar_datos()
    print(f"Dataset: {dtf.shape[0]} filas | rango: "
          f"{dtf.index.min().date()} → {dtf.index.max().date()}\n")
    print(f"Fechas de corte evaluadas: {FECHAS_CORTE}\n")
 
    df_det, df_prom = backtest_multifecha(dtf, column="MW")
 
    print("── Resultados por fecha ─────────────────────────────────")
    print(df_det.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
 
    print("\n── Promedio sobre las 6 fechas ──────────────────────────")
    print(df_prom.to_string(float_format=lambda x: f"{x:.2f}"))
 
    print("\nGenerando gráfico de comparación...")
    ruta = graficar_multifecha(dtf, column="MW", guardar="backtest_multifecha.png")
    print(f"Gráfico guardado en {ruta}")
