
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

SEED = 168
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
#PARÁMETROS EXPERIMENTALES
# ---------------------------------------------------------------------------
TASA_BASE = 0.50

# Escenario hipotético: 1 = mensaje que aparenta proceder de una autoridad.
# 0 = escenario sin ese contexto específico.
ESCENARIO_AUTORIDAD = 0

# Betas asociados a un aumento de una desviación estándar en cada predictor.
# Son valores iniciales de diseño para el experimento, NO efectos demostrados.
BETAS = {
    "edad": 0.00,
    "edad2": 0.00,
    "nivel_educativo": -0.25,
    "alfabetizacion": -0.40,
    "uso_redes": 0.20,
    "exp_content": 0.30,
    "interaccion": 0.00,
    "impulsividad": 0.30,
    "estres_ecn": 0.25,
    "confianza_aut": 0.00,
    "confianza_aut_x_escenario": 0.30,
}

MAP_EDUC = {
    "Sin estudios": 0,
    "Básica": 1,
    "Media superior": 2,
    "Superior": 3,
}
MAP_REDES = {
    "Nunca": 0,
    "Ocasional": 1,
    "Semanal": 2,
    "Diario": 3,
    "Intensivo": 4,
}

def estandarizar(x):

    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - x.mean()) / sd


def construir_X(df, escenario=ESCENARIO_AUTORIDAD):

    requeridas = [
        "edad", "nivel_educativo", "alfabetizacion", "uso_redes",
        "exp_content", "interaccion", "impulsividad", "estres_ecn",
        "confianza_aut",
    ]
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    educ = df["nivel_educativo"].map(MAP_EDUC)
    redes = df["uso_redes"].map(MAP_REDES)
    if educ.isna().any():
        desconocidos = df.loc[educ.isna(), "nivel_educativo"].unique().tolist()
        raise ValueError(f"Categorías educativas no reconocidas: {desconocidos}")
    if redes.isna().any():
        desconocidos = df.loc[redes.isna(), "uso_redes"].unique().tolist()
        raise ValueError(f"Categorías de uso de redes no reconocidas: {desconocidos}")

    z_edad = estandarizar(df["edad"])
    X = pd.DataFrame(index=df.index)
    X["edad"] = z_edad
    X["edad2"] = estandarizar(z_edad ** 2)
    X["nivel_educativo"] = estandarizar(educ)
    X["alfabetizacion"] = estandarizar(df["alfabetizacion"])
    X["uso_redes"] = estandarizar(redes)
    X["exp_content"] = estandarizar(df["exp_content"])
    X["interaccion"] = estandarizar(df["interaccion"])
    X["impulsividad"] = estandarizar(df["impulsividad"])
    X["estres_ecn"] = estandarizar(df["estres_ecn"])
    X["confianza_aut"] = estandarizar(df["confianza_aut"])
    X["confianza_aut_x_escenario"] = X["confianza_aut"] * int(bool(escenario))
    return X

def lineal(X, betas):
    faltantes = [col for col in X.columns if col not in betas]
    if faltantes:
        raise ValueError(f"Faltan coeficientes para: {faltantes}")
    return X.to_numpy() @ np.array([betas[col] for col in X.columns], dtype=float)


def calibrar_b0(eta, tasa_base):
    if not 0 < tasa_base < 1:
        raise ValueError("TASA_BASE debe estar estrictamente entre 0 y 1.")
    eta = np.asarray(eta, dtype=float)
    funcion = lambda b0: expit(b0 + eta).mean() - tasa_base
    return brentq(funcion, -30, 30)


def susceptibilidad(df, betas=BETAS, tasa_base=TASA_BASE,
                    escenario=ESCENARIO_AUTORIDAD):
    X = construir_X(df, escenario)
    eta = lineal(X, betas)
    b0 = calibrar_b0(eta, tasa_base)
    p = expit(b0 + eta)
    return p, b0, X

def sensibilidad(df, factor=0.5, delta_cero=0.15,
                 tasa_base=TASA_BASE, escenario=ESCENARIO_AUTORIDAD):

    X = construir_X(df, escenario)
    eta_base = lineal(X, BETAS)
    b0_base = calibrar_b0(eta_base, tasa_base)
    p_base = expit(b0_base + eta_base)
    umbral = 0.5
    metrica_base = float((p_base > umbral).mean())

    filas = []
    for variable, valor in BETAS.items():
        delta = abs(valor) * factor if valor != 0 else delta_cero
        proporciones = []
        for signo in (-1, 1):
            betas_alterados = dict(BETAS)
            betas_alterados[variable] = valor + signo * delta
            eta = lineal(X, betas_alterados)
            b0 = calibrar_b0(eta, tasa_base)
            p = expit(b0 + eta)
            proporciones.append(float((p > umbral).mean()))
        filas.append({
            "variable": variable,
            "beta_original": valor,
            "beta_bajo": valor - delta,
            "beta_alto": valor + delta,
            "proporcion_bajo": proporciones[0],
            "proporcion_alto": proporciones[1],
            "rango_proporcion": abs(proporciones[1] - proporciones[0]),
        })

    resultado = pd.DataFrame(filas).sort_values(
        "rango_proporcion", ascending=False
    )
    return metrica_base, resultado

if __name__ == "__main__":
    archivo_entrada = "dataset_simulado.csv"
    archivo_salida = "dataset_con_susceptibilidad0.csv"

    df = pd.read_csv(archivo_entrada)

    p, b0, X = susceptibilidad(df)
    df["p_susceptibilidad"] = p

    etiqueta_binaria = rng.binomial(1, p)
    df["susceptibilidad"] = np.where(etiqueta_binaria == 1, 0, 1)

    print(f"Intercepto calibrado beta_0 = {b0:.3f}")
    print(f"Probabilidad media = {p.mean():.3f}")
    print(
        "Proporción simulada de clase 1 = "
        f"{(df['susceptibilidad'] == 1).mean():.3f}"
    )
    print("\nProbabilidad media por grupo de edad:")
    if "grupo_edad" in df.columns:
        print(
            df.groupby("grupo_edad", observed=True)["p_susceptibilidad"]
            .mean().round(3)
        )
    else:
        print("No existe la columna 'grupo_edad'; se omite este resumen.")

    print("\nProbabilidad media por nivel educativo:")
    print(
        df.groupby("nivel_educativo", observed=True)["p_susceptibilidad"]
        .mean().round(3)
    ) 
    try:
        import statsmodels.api as sm

        y_binaria = (df["susceptibilidad"] == 1).astype(int)
        modelo = sm.Logit(y_binaria, sm.add_constant(X)).fit(disp=0)
        comparacion = pd.DataFrame({
            "beta_asignado": pd.Series(BETAS),
            "beta_estimado": modelo.params.drop("const"),
        }).round(3)
        print(f"\nRecuperación aproximada de parámetros (n={len(df)}):")
        print(comparacion)
    except ImportError:
        print("\nPara la comprobación de coeficientes instala statsmodels.")
    except Exception as error:
        print(f"\nNo se pudo ajustar la regresión de comprobación: {error}")

    base, tornado = sensibilidad(df)
    print(
        "\nSensibilidad: proporción de perfiles por encima del umbral "
        f"experimental (base = {base:.3f})"
    )
    print(tornado.round(3).to_string(index=False))

    df.to_csv(archivo_salida, index=False)
    print(f"\nDataset guardado en: {archivo_salida}")
