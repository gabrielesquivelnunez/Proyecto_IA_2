# =============================================================================
# main3.py
# -----------------------------------------------------------------------------
# Agente de predicción de series de tiempo eléctricas.
#
# Modificacion del main2.py de ErickVM7/Proyecto-Electrico (agente LLM local con Ollama,
# enfoque de Mauro Di Pietro). Para poder agregar dos
# herramientas nuevas como aporte del Proyecto 2 (IE0435, I-2026):
#
#     predict_knn   -> predicción con K-Nearest Neighbors (regresión)
#     predict_tree  -> predicción con árboles de regresión
#
# Las herramientas viven en herramientas_ml.py y aquí se registran en el flujo
# del agente (dic_tools, available_tools, normalizadores y prompt).
# =============================================================================

# =============================
# Librerías
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import ast, io, contextlib, re, json, subprocess
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import ollama
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Herramientas nuevas (aporte de este proyecto)
from herramientas_ml import (
    predict_knn as _predict_knn,
    predict_tree as _predict_tree,
    tool_predict_knn,
    tool_predict_tree,
)

# =============================
# Configuración del modelo
# =============================
llm = "qwen2.5:7b"
try:
    subprocess.run(["ollama", "pull", llm], check=False)
    print(f" Modelo cargado: {llm}")
except FileNotFoundError:
    print("  Ollama no está instalado o no está en el PATH. "
          "Las herramientas de predicción funcionan igual; el chat con el "
          "LLM requiere Ollama (https://ollama.com/download).")

# Carpeta donde se guardan las predicciones
save_dir = "predicciones_generadas"

# =============================
# Carga directa del dataset
# =============================
DATA_PATH = "datos_limpios.csv"

try:
    dtf = pd.read_csv(DATA_PATH)
    datetime_col = next((col for col in dtf.columns
                         if any(x in col.lower() for x in ["fecha", "date", "hora"])), None)
    if datetime_col is None:
        raise ValueError("No se encontró columna de fecha/hora.")

    dtf[datetime_col] = pd.to_datetime(dtf[datetime_col], errors="coerce")
    dtf.set_index(datetime_col, inplace=True)
    dtf.sort_index(inplace=True)
    dtf = dtf.asfreq("15min")

    print(f" Dataset '{DATA_PATH}' cargado correctamente ({len(dtf)} filas)")
    print(f"   Índice temporal: {datetime_col} (frecuencia 15 min)")
    print(f"   Columnas disponibles: {list(dtf.columns)}\n")
    print(" Primeras filas del dataset:")
    print(dtf.head(), "\n")

except Exception as e:
    print(f" Error al cargar {DATA_PATH}: {e}")
    exit()


# =============================
# Herramientas base (de Erick)
# =============================
def final_answer(text: str) -> str:
    return text


tool_final_answer = {
    'type': 'function',
    'function': {
        'name': 'final_answer',
        'description': 'Devuelve una respuesta en lenguaje natural al usuario',
        'parameters': {
            'type': 'object', 'required': ['text'],
            'properties': {'text': {'type': 'string',
                                    'description': 'respuesta en lenguaje natural'}}
        }
    }
}


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def sanitize_code(code: str) -> str:
    code = code.replace("df[", "dtf[")
    if code.count("(") > code.count(")"):
        code += ")" * (code.count("(") - code.count(")"))
    if code.count("[") > code.count("]"):
        code += "]" * (code.count("[") - code.count("]"))
    if code.count('"') % 2 != 0:
        code += '"'
    if code.count("'") % 2 != 0:
        code += "'"
    return code


def code_exec(code: str) -> str:
    output = io.StringIO()
    code = sanitize_code(code.strip())
    if not code.startswith("print(") or not code.endswith(")"):
        return "Error: usa print(...) para mostrar resultados."
    if not is_valid_python(code):
        return "Error: código incompleto o inválido."
    with contextlib.redirect_stdout(output):
        try:
            exec(code, globals())
        except Exception as e:
            print(f"Error: {e}")
    return output.getvalue()


tool_code_exec = {
    'type': 'function',
    'function': {
        'name': 'code_exec',
        'description': 'Ejecuta código Python. Siempre usar print() para mostrar la salida.',
        'parameters': {
            'type': 'object', 'required': ['code'],
            'properties': {'code': {'type': 'str',
                                    'description': 'código Python a ejecutar'}}
        }
    }
}


def normalize_plot_args(t_inputs):
    if not isinstance(t_inputs, dict):
        return t_inputs
    if "columns" in t_inputs:
        if isinstance(t_inputs["columns"], str):
            try:
                t_inputs["columns"] = json.loads(t_inputs["columns"].replace("'", '"'))
            except Exception:
                t_inputs["columns"] = [t_inputs["columns"]]
        elif t_inputs["columns"] is None:
            t_inputs["columns"] = []
    return t_inputs


def plot_data(columns=None, start_date=None, end_date=None, title="Gráfico de datos"):
    try:
        df = dtf.copy()
        if columns:
            cols_validas = [c for c in columns if c in df.columns]
            df = df[cols_validas]
        if start_date and end_date:
            df = df.loc[start_date:end_date]
        elif start_date:
            df = df.loc[start_date]
        df.plot(figsize=(12, 5), linestyle="--")
        plt.title(title)
        plt.xlabel("Tiempo (15 min)")
        plt.ylabel("Potencia [MW]")
        plt.grid(True)
        plt.show()
        return f"Gráfico generado con columnas {columns or list(df.columns)}."
    except Exception as e:
        return f"Error al graficar: {e}"


tool_plot_data = {
    'type': 'function',
    'function': {
        'name': 'plot_data',
        'description': 'Genera gráficos de columnas del dataset en un rango de fechas.',
        'parameters': {
            'type': 'object', 'required': ['columns'],
            'properties': {
                'columns': {'type': 'array', 'description': 'columnas a graficar'},
                'start_date': {'type': 'string', 'description': 'fecha inicial'},
                'end_date': {'type': 'string', 'description': 'fecha final'},
            }
        }
    }
}


def predict_data(model="prophet", column=None, horizon=None, end_date=None):
    """Predicción con Prophet o SARIMA (herramienta original de Erick)."""
    try:
        df = dtf.copy()
        if column is None or column not in df.columns:
            return f"Error: columna inválida. Disponibles: {list(df.columns)}"

        df = df[[column]].dropna().reset_index()
        df.columns = ["ds", "y"]
        freq = "15min"
        last_date = df["ds"].max()

        steps = 96
        if horizon:
            if isinstance(horizon, str):
                nums = re.findall(r"\d+", horizon)
                days = int(nums[0]) if nums else 1
                steps = max(1, days * 96)
            elif isinstance(horizon, int):
                steps = max(1, horizon * 96)
        if end_date and (not horizon or isinstance(horizon, str)
                         and re.match(r"^\d{4}-\d{2}-\d{2}$", horizon)):
            try:
                target_date = pd.to_datetime(end_date)
                delta = target_date - last_date
                if delta.total_seconds() <= 0:
                    return f"La fecha {end_date} ya está incluida en los datos."
                steps = max(1, int(delta.total_seconds() / (15 * 60)))
            except Exception as e:
                return f"Error interpretando end_date: {e}"

        model = model.lower()

        if model == "prophet":
            m = Prophet(daily_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=steps, freq=freq)
            forecast = m.predict(future)
            forecast_pred = forecast[forecast["ds"] > last_date][["ds", "yhat"]]
            forecast_pred = forecast_pred.rename(columns={"ds": "fecha", "yhat": "MW_pred"})
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_pred["fecha"], forecast_pred["MW_pred"], "o-",
                     label="Predicción (Prophet)")
            plt.title(f"Predicción Prophet para {column} ({steps} pasos)")
            plt.xlabel("Fecha"); plt.ylabel(column); plt.grid(True); plt.legend(); plt.show()
            out_df = forecast_pred.copy()

        elif model == "arima":
            df_use = df.tail(96 * 30) if len(df) > 10000 else df.copy()
            df_use.set_index("ds", inplace=True)
            sarimax = SARIMAX(df_use["y"], order=(2, 1, 2),
                              seasonal_order=(1, 0, 1, 96)).fit(disp=False)
            future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]
            forecast_vals = sarimax.forecast(steps=steps)
            out_df = pd.DataFrame({"fecha": future_dates, "MW_pred": forecast_vals})
            plt.figure(figsize=(10, 5))
            plt.plot(out_df["fecha"], out_df["MW_pred"], "o-", label="Predicción (SARIMA)")
            plt.title(f"Predicción SARIMA para {column} ({steps} pasos)")
            plt.xlabel("Fecha"); plt.ylabel(column); plt.grid(True); plt.legend(); plt.show()
        else:
            return "Error: modelo no reconocido. Usa 'prophet' o 'arima'."

        os.makedirs(save_dir, exist_ok=True)
        start_str = pd.to_datetime(out_df["fecha"].min()).strftime("%Y%m%d_%H%M")
        end_str = pd.to_datetime(out_df["fecha"].max()).strftime("%Y%m%d_%H%M")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pred_{model}_{column}_{start_str}_to_{end_str}_{timestamp}.csv"
        out_path = os.path.join(save_dir, filename)
        out_df[["fecha", "MW_pred"]].to_csv(out_path, index=False)
        print(f"\n💾 Guardado en: {out_path}")
        return f"Predicción {model.upper()} completada ({len(out_df)} puntos). Archivo: {out_path}"

    except Exception as e:
        return f"Error durante la predicción: {e}"


tool_predict_data = {
    'type': 'function',
    'function': {
        'name': 'predict_data',
        'description': 'Predicción de series de tiempo con Prophet o ARIMA.',
        'parameters': {
            'type': 'object', 'required': ['model', 'column'],
            'properties': {
                'model': {'type': 'string', 'description': '"prophet" o "arima"'},
                'column': {'type': 'string', 'description': 'columna a predecir'},
                'horizon': {'type': 'integer', 'description': 'horizonte en días'},
            }
        }
    }
}


# =============================
# Wrappers de las herramientas nuevas
# -----------------------------------------------------------------------------
# herramientas_ml.predict_knn / predict_tree reciben `dtf` como primer
# argumento; aquí lo inyectamos desde el global para que la firma vista por el
# agente sea idéntica a predict_data (sin pasar el dataframe).
# =============================
def predict_knn(column=None, horizon=None, n_neighbors=15):
    return _predict_knn(dtf, column=column, horizon=horizon,
                        n_neighbors=n_neighbors, save_dir=save_dir)


def predict_tree(column=None, horizon=None, max_depth=12):
    return _predict_tree(dtf, column=column, horizon=horizon,
                         max_depth=max_depth, save_dir=save_dir)


# =============================
# Diccionario de herramientas
# =============================
dic_tools = {
    "final_answer": final_answer,
    "code_exec": code_exec,
    "plot_data": plot_data,
    "predict_data": predict_data,
    "predict_knn": predict_knn,     # NUEVA
    "predict_tree": predict_tree,   # NUEVA
}


# =============================
# Normalizadores de argumentos
# =============================
def normalize_predict_args(t_inputs):
    if not isinstance(t_inputs, dict):
        return t_inputs
    for k in ["forecast_type", "prediction_type", "days_ahead"]:
        t_inputs.pop(k, None)
    if "horizon" in t_inputs and isinstance(t_inputs["horizon"], str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", t_inputs["horizon"]):
            t_inputs["end_date"] = t_inputs.pop("horizon")
        else:
            num = re.findall(r"\d+", t_inputs["horizon"])
            t_inputs["horizon"] = int(num[0]) if num else 1
    if "date" in t_inputs and "end_date" not in t_inputs:
        t_inputs["end_date"] = t_inputs.pop("date")
    if "horizon" not in t_inputs:
        t_inputs["horizon"] = 1
    if "model" in t_inputs and isinstance(t_inputs["model"], str):
        t_inputs["model"] = t_inputs["model"].lower()
    return t_inputs


def normalize_ml_args(t_inputs):
    """Normaliza argumentos para predict_knn / predict_tree."""
    if not isinstance(t_inputs, dict):
        return t_inputs
    # horizon en texto -> días (int)
    if "horizon" in t_inputs and isinstance(t_inputs["horizon"], str):
        num = re.findall(r"\d+", t_inputs["horizon"])
        t_inputs["horizon"] = int(num[0]) if num else 1
    if "horizon" not in t_inputs:
        t_inputs["horizon"] = 1
    # columna válida por defecto
    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
        t_inputs["column"] = list(dtf.columns)[0]
    return t_inputs


# =============================
# Ejecutor de herramientas
# =============================
def use_tool(agent_res: dict, dic_tools: dict) -> dict:
    msg = agent_res["message"]
    res, t_name, t_inputs = "", "", ""

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            t_name = tool["function"]["name"]
            raw_args = tool["function"]["arguments"]
            try:
                t_inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                t_inputs = raw_args

            if t_name == "plot_data":
                t_inputs = normalize_plot_args(t_inputs)
            elif t_name == "code_exec":
                if isinstance(t_inputs, dict):
                    t_inputs = {"code": t_inputs.get("code", "")}
            elif t_name == "predict_data":
                t_inputs = normalize_predict_args(t_inputs)
                if isinstance(t_inputs, dict):
                    t_inputs.setdefault("model", "prophet")
                    t_inputs.setdefault("horizon", 1)
                    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
                        t_inputs["column"] = list(dtf.columns)[0]
            elif t_name in ("predict_knn", "predict_tree"):
                t_inputs = normalize_ml_args(t_inputs)
            elif t_name == "final_answer" and isinstance(t_inputs, dict) and "final_answer" in t_inputs:
                t_inputs = {"text": t_inputs["final_answer"]}

            if f := dic_tools.get(t_name):
                print(f"🔧 > {t_name} -> Inputs: {t_inputs}")
                try:
                    t_output = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                except Exception as e:
                    cols = list(dtf.columns) if 'dtf' in globals() else 'No hay dataset'
                    t_output = f"Error ejecutando {t_name}: {e}. Columnas: {cols}"
                print(f" Resultado:\n{t_output}\n")
                res = t_output
            else:
                print(f" > {t_name} -> NotFound")

    elif msg.get("content", "") and msg["content"].strip().startswith("{"):
        try:
            tool_call = json.loads(msg["content"])
            t_name = tool_call.get("name", "")
            t_inputs = tool_call.get("arguments", {})
            if t_name == "predict_data":
                t_inputs = normalize_predict_args(t_inputs)
            elif t_name in ("predict_knn", "predict_tree"):
                t_inputs = normalize_ml_args(t_inputs)
            if f := dic_tools.get(t_name):
                print(f" > {t_name} -> Inputs: {t_inputs}")
                res = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                print(f" Resultado:\n{res}\n")
            else:
                res = f"Herramienta {t_name} no encontrada."
        except Exception as e:
            res = f" Error al interpretar JSON: {e}\nContenido: {msg.get('content')}"

    elif msg.get("content", ""):
        res = msg["content"]
        print(f" {res}")

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}


def run_agent(llm, messages, available_tools):
    tool_used, local_memory = '', ''
    used_compute = False
    while tool_used != 'final_answer':
        try:
            agent_res = ollama.chat(model=llm, messages=messages,
                                    tools=[v for v in available_tools.values()])
            dic_res = use_tool(agent_res, dic_tools)
            res, tool_used, inputs_used = dic_res["res"], dic_res["tool_used"], dic_res["inputs_used"]

            if tool_used in ("code_exec", "plot_data"):
                used_compute = True

            user_query = messages[-1]["content"].lower()
            needs_compute = any(word in user_query for word in [
                "promedio", "media", "máximo", "mínimo", "suma", "resta",
                "gráfico", "grafica", "plot", "visualiza", "filtra",
                "porcentaje", "calcula", "valor", "estadística", "histograma", "error",
            ])
            if tool_used == "final_answer" and needs_compute and not used_compute:
                print(" > El modelo intentó responder sin calcular. Reintentando...")
                messages.append({"role": "user",
                                 "content": "Debes usar code_exec o plot_data antes de final_answer."})
                tool_used = ""
                continue
        except Exception as e:
            print(" >", e)
            res = f"Intenté usar {tool_used} pero falló. Intentaré otra cosa."
            messages.append({"role": "assistant", "content": res})

        if tool_used not in ['', 'final_answer']:
            local_memory += f"\nTool used: {tool_used}.\nInput: {inputs_used}.\nOutput: {res}"
            messages.append({"role": "assistant", "content": f"Resultado: {res}"})
            available_tools.pop(tool_used, None)
            if len(available_tools) == 1:
                messages.append({"role": "user", "content": "ahora activa la herramienta final_answer."})
        if tool_used == '':
            break
    return res


# =============================
# Prompt del sistema
# =============================
prompt = """
Eres un Analista de Datos especializado en series de tiempo eléctricas.

Contexto del dataset:
- El archivo `datos_limpios.csv` ya está cargado en memoria como `dtf`.
- Datos reales de consumo eléctrico nacional con frecuencia de 15 minutos.
- Columnas:
  • `MW`: potencia eléctrica medida (megavatios).
  • `MW_P`: potencia programada/predicha por el ICE (megavatios).
- Índice temporal `fechaHora` en formato datetime (intervalos de 15 min).

Herramientas disponibles:
1. code_exec   -> ejecuta código Python con print() (cálculos, estadísticas).
2. plot_data   -> genera gráficos de columnas en un rango de fechas.
3. predict_data-> predicción con Prophet o ARIMA.
4. predict_knn -> predicción con K-Nearest Neighbors (regresión).
5. predict_tree-> predicción con árbol de regresión.
6. final_answer-> responde en lenguaje natural.

Guía para elegir modelo de predicción:
- Prophet/ARIMA: buenos para tendencia y estacionalidad de largo plazo.
- KNN: bueno cuando los días futuros se parecen a días pasados ya vistos.
- Árbol de regresión: rápido e interpretable, capta reglas hora/día.
Si el usuario no especifica el modelo, puedes sugerir uno y explicar por qué.

Restricciones:
- No cargues ni reemplaces el dataset (ya está como `dtf`).
- No inventes columnas ni archivos. Usa los nombres reales de columnas.
- Usa comillas dobles en columnas, p.ej. dtf["MW"].
- Si piden una fecha que no existe, informa el error con final_answer.

Ejemplos:
- "promedio de MW" -> code_exec con print(dtf["MW"].mean()).
- "predice MW 2 días con KNN" -> predict_knn {"column":"MW","horizon":2}.
- "predice MW mañana con un árbol" -> predict_tree {"column":"MW","horizon":1}.
"""


# =============================
# Bucle principal
# =============================
if __name__ == "__main__":
    messages = [{"role": "system", "content": prompt}]
    print(" Agente listo (Prophet/ARIMA/KNN/Árbol). Escribe 'quit' para salir.\n")

    while True:
        q = input(" > ")
        if q.lower() == "quit":
            break
        messages.append({"role": "user", "content": q})
        available_tools = {
            "final_answer": tool_final_answer,
            "code_exec": tool_code_exec,
            "plot_data": tool_plot_data,
            "predict_data": tool_predict_data,
            "predict_knn": tool_predict_knn,
            "predict_tree": tool_predict_tree,
        }
        res = ollama.chat(model=llm, messages=messages,
                          tools=[v for v in available_tools.values()], format="json")
        dic_res = use_tool(res, dic_tools)
        print("👽 >", dic_res["res"], "\n")
        messages.append({"role": "assistant", "content": dic_res["res"]})
