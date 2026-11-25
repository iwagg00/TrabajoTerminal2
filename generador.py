import numpy as np
from scipy.stats import truncnorm

SEED = 192
rng = np.random.default_rng(SEED)

def _muestreo_categorico_por_edad(edad, tabla_probs, categorias, bins):

    n = len(edad)
    resultado = np.empty(n, dtype=object)

    idx_bin = np.full(n, -1, dtype=int)
    for i, (e_min, e_max) in enumerate(bins):
        mask = (edad >= e_min) & (edad <= e_max)
        idx_bin[mask] = i

    idx_bin[edad < bins[0][0]] = 0
    idx_bin[edad > bins[-1][1]] = len(bins) - 1

    for i in range(len(bins)):
        mask = idx_bin == i
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        resultado[mask] = rng.choice(
            categorias,
            size=n_bin,
            p=tabla_probs[i]
        )

    return resultado


# ---------------------------------------------------------------------------
# Funciones de generación de datos
# ---------------------------------------------------------------------------
def gen_edad(n):
    mu = 35
    sigma = 12
    min_edad = 18
    max_edad = 70

    a = (min_edad - mu) / sigma
    b = (max_edad - mu) / sigma

    edad = truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )

    return np.rint(edad).astype(int)


def gen_nivEsc(edad):
    categorias = ["Sin estudios", "Básica", "Media superior", "Superior"]

    bins = [
        (18, 25),
        (26, 35),
        (36, 50),
        (51, 70),
    ]

    tabla_probs = np.array([
        [0.02, 0.15, 0.38, 0.45],  # 18-25
        [0.03, 0.20, 0.35, 0.42],  # 26-35
        [0.05, 0.30, 0.35, 0.30],  # 36-50
        [0.10, 0.40, 0.30, 0.20],  # 51-70
    ])

    return _muestreo_categorico_por_edad(edad, tabla_probs, categorias, bins)


def gen_alf(n):
    mu = 0.55
    sigma = 0.18
    minimo = 0
    maximo = 1

    a = (minimo - mu) / sigma
    b = (maximo - mu) / sigma

    return truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )


def gen_usoRedes(edad):
    categorias = ["Nunca", "Ocasional", "Semanal", "Diario", "Intensivo"]

    bins = [
        (18, 25),
        (26, 35),
        (36, 50),
        (51, 70),
    ]

    tabla_probs = np.array([
        [0.02, 0.05, 0.13, 0.40, 0.40],  # 18-25
        [0.03, 0.07, 0.15, 0.45, 0.30],  # 26-35
        [0.05, 0.10, 0.20, 0.45, 0.20],  # 36-50
        [0.12, 0.18, 0.25, 0.35, 0.10],  # 51-70
    ])

    return _muestreo_categorico_por_edad(edad, tabla_probs, categorias, bins)


def gen_expContent(n):
    mu = 0.50
    sigma = 0.20
    minimo = 0
    maximo = 1

    a = (minimo - mu) / sigma
    b = (maximo - mu) / sigma

    return truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )


def gen_interaccion(n):
    mu = 0.50
    sigma = 0.15
    minimo = 0
    maximo = 1

    a = (minimo - mu) / sigma
    b = (maximo - mu) / sigma

    return truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )


def gen_impulsividad(n):
    mu = 2.85
    sigma = 0.74
    minimo = 1
    maximo = 5

    a = (minimo - mu) / sigma
    b = (maximo - mu) / sigma

    return truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )


def gen_estresEcn(n):
    mu = 0.60
    sigma = 0.20
    minimo = 0
    maximo = 1

    a = (minimo - mu) / sigma
    b = (maximo - mu) / sigma

    return truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )


def gen_ConfianzaAut(n):
    mu = 3.20
    sigma = 0.80
    minimo = 1
    maximo = 5

    a = (minimo - mu) / sigma
    b = (maximo - mu) / sigma

    return truncnorm.rvs(
        a, b,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=rng
    )


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    n = 20_000

    edad = gen_edad(n)
    nivel_educativo = gen_nivEsc(edad)
    uso_redes = gen_usoRedes(edad)

    alfabetizacion = gen_alf(n)
    exp_content = gen_expContent(n)
    interaccion = gen_interaccion(n)
    impulsividad = gen_impulsividad(n)
    estres_ecn = gen_estresEcn(n)
    confianza_aut = gen_ConfianzaAut(n)

    df = pd.DataFrame({
        "edad": edad,
        "nivel_educativo": nivel_educativo,
        "uso_redes": uso_redes,
        "alfabetizacion": alfabetizacion,
        "exp_content": exp_content,
        "interaccion": interaccion,
        "impulsividad": impulsividad,
        "estres_ecn": estres_ecn,
        "confianza_aut": confianza_aut,
    })

    df["grupo_edad"] = pd.cut(
        df["edad"],
        bins=[17, 25, 35, 50, 70],
        labels=["18-25", "26-35", "36-50", "51-70"]
    )

    print("Primeras filas del dataset:")
    print(df.head(10))

    print("\nProporciones nivel educativo por grupo de edad:")
    print(pd.crosstab(df["grupo_edad"], df["nivel_educativo"], normalize="index").round(2))

    print("\nProporciones uso de redes por grupo de edad:")
    print(pd.crosstab(df["grupo_edad"], df["uso_redes"], normalize="index").round(2))

    columnas_numericas = [
        "edad", "alfabetizacion", "exp_content",
        "interaccion", "impulsividad", "estres_ecn", "confianza_aut"
    ]
    print("\nEstadísticas descriptivas:")
    print(df[columnas_numericas].describe().round(3))

    variables_continuas = [
        "alfabetizacion", "exp_content", "interaccion",
        "impulsividad", "estres_ecn", "confianza_aut"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, variables_continuas):
        ax.hist(df[col], bins=30, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
    fig.suptitle("Distribuciones de las variables generadas")
    fig.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(df["edad"], bins=53)
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de edad")
    plt.show()

    df.to_csv("dataset_simulado.csv", index=False)
    print("\nDataset exportado a 'dataset_simulado.csv' con", df.shape[0],
          "filas y", df.shape[1], "columnas.")