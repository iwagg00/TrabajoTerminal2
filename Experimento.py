#!/usr/bin/env python3
import os
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN: modifica estos valores directamente aquí
# ============================================================
ARCHIVO_CSV = "dataset.csv"       # Ruta del conjunto de datos
CARPETA_SALIDA = "resultados"     # Carpeta para reportes, gráficas y modelos
K_FOLDS = 5
REPETICIONES = 5
SEMILLA = 42
USAR_KERAS = True                 # Si no hay TensorFlow, se usa MLPClassifier
AGREGAR_INTERACCION = False       # confianza_aut * escenario
UMBRAL = 0.5

OBJETIVO = "susceptibilidad"
EXCLUIR = ["grupo_edad", "p_susceptibilidad", OBJETIVO]
MAP_EDUC = {"Sin estudios": 0, "Básica": 1, "Media superior": 2, "Superior": 3}
MAP_REDES = {"Nunca": 0, "Ocasional": 1, "Semanal": 2, "Diario": 3, "Intensivo": 4}
METRICAS = ["accuracy", "precision", "recall", "f1", "auc_roc"]

BETAS_ASIGNADOS = {
    "nivel_educativo": -0.25, "alfabetizacion": -0.40, "uso_redes": 0.20,
    "exp_content": 0.30, "interaccion": 0.00, "impulsividad": 0.30,
    "estres_ecn": 0.25, "confianza_aut": 0.00, "edad": 0.00,
    "confianza_aut_x_escenario": 0.30,
}


def cargar(ruta, con_interaccion):
    df = pd.read_csv(ruta)
    y = (df[OBJETIVO] == 1).astype(int).to_numpy()
    p_real = df["p_susceptibilidad"].to_numpy() if "p_susceptibilidad" in df else None

    X = df.drop(columns=[c for c in EXCLUIR if c in df.columns]).copy()
    X["nivel_educativo"] = X["nivel_educativo"].map(MAP_EDUC)
    X["uso_redes"] = X["uso_redes"].map(MAP_REDES)

    if X.isna().any().any():
        raise ValueError(
            "Hay valores faltantes o categorías no reconocidas:\n"
            f"{X.isna().sum()[X.isna().sum() > 0]}"
        )

    if con_interaccion and "escenario" in X.columns:
        X["confianza_aut_x_escenario"] = X["confianza_aut"] * X["escenario"]

    return X, y, p_real


class MLPKeras:
    def __init__(self, seed, epocas=100, lote=64, lr=1e-3):
        self.seed = seed
        self.epocas = epocas
        self.lote = lote
        self.lr = lr

    def fit(self, X, y):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        Xa, Xb, ya, yb = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=self.seed
        )
        L = tf.keras.layers
        self.m = tf.keras.Sequential([
            L.Input(shape=(X.shape[1],)),
            L.Dense(32, activation="relu"),
            L.Dropout(0.2),
            L.Dense(16, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ])
        self.m.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="binary_crossentropy"
        )
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        self.m.fit(
            Xa, ya, validation_data=(Xb, yb), epochs=self.epocas,
            batch_size=self.lote, callbacks=[es], verbose=0
        )
        return self

    def predict_proba(self, X):
        return self.m.predict(X, verbose=0, batch_size=2048).ravel()


def construir_modelos(seed, usar_keras):
    if usar_keras:
        mlp = MLPKeras(seed)
    else:
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16), early_stopping=True,
            max_iter=300, random_state=seed
        )

    return {
        "Naive Bayes": GaussianNB(),
        "Regresión logística": LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=seed
        ),
        "MLP": mlp,
    }


def prob(modelo, X):
    p = modelo.predict_proba(X)
    return p if p.ndim == 1 else p[:, 1]


def metricas(y, p):
    pred = (p >= UMBRAL).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, p),
    }


def experimento(X, y):
    nombres = list(X.columns)
    Xv = X.to_numpy(float)
    n = len(y)
    filas, coefs, perm = [], [], []
    oof = {}
    rng_perm = np.random.default_rng(SEMILLA)

    for rep in range(REPETICIONES):
        semilla = SEMILLA + rep
        skf = StratifiedKFold(
            n_splits=K_FOLDS, shuffle=True, random_state=semilla
        )

        for fold, (tr, va) in enumerate(skf.split(Xv, y), start=1):
            sc = StandardScaler().fit(Xv[tr])
            Xtr, Xva = sc.transform(Xv[tr]), sc.transform(Xv[va])

            print(
                f"Repetición {rep + 1}/{REPETICIONES} | "
                f"fold {fold}/{K_FOLDS}",
                end="\r", flush=True
            )

            for nombre, modelo in construir_modelos(semilla, USAR_KERAS).items():
                modelo.fit(Xtr, y[tr])
                p = prob(modelo, Xva)
                filas.append({
                    "modelo": nombre, "repeticion": rep + 1, "fold": fold,
                    **metricas(y[va], p)
                })

                if rep == 0:
                    oof.setdefault(nombre, np.zeros(n))[va] = p
                    auc_base = roc_auc_score(y[va], p)

                    for j, var in enumerate(nombres):
                        Xp = Xva.copy()
                        Xp[:, j] = rng_perm.permutation(Xp[:, j])
                        perm.append({
                            "modelo": nombre, "variable": var, "fold": fold,
                            "caida_auc": auc_base - roc_auc_score(
                                y[va], prob(modelo, Xp)
                            )
                        })

                    if nombre == "Regresión logística":
                        coefs.append(modelo.coef_[0].copy())

    print(" " * 80)
    return pd.DataFrame(filas), np.array(coefs), pd.DataFrame(perm), oof


def resumen(res, y, p_real):
    g = res.groupby("modelo")[METRICAS]
    media, desv = g.mean(), g.std()
    tabla = media.copy()

    for m in METRICAS:
        tabla[m] = [
            f"{a:.3f} ± {b:.3f}" for a, b in zip(media[m], desv[m])
        ]

    if p_real is not None:
        ref = metricas(y, p_real)
        tabla.loc["(Referencia) Oráculo: p real"] = [
            f"{ref[m]:.3f}" for m in METRICAS
        ]

    return tabla, media


def tabla_coeficientes(coefs, nombres, con_escenario):
    c = pd.DataFrame({
        "variable": nombres,
        "coef_medio": coefs.mean(axis=0),
        "coef_std": coefs.std(axis=0, ddof=1) if len(coefs) > 1 else 0.0,
    })
    c["odds_ratio"] = np.exp(c["coef_medio"])
    asignados = dict(BETAS_ASIGNADOS)

    if not con_escenario:
        asignados["confianza_aut"] += asignados["confianza_aut_x_escenario"]

    c["beta_asignado"] = c["variable"].map(asignados)
    return c.reindex(c["coef_medio"].abs().sort_values(ascending=False).index)


def graficas(res, oof, y, p_real, coef_df, perm_df, carpeta):
    modelos = list(res["modelo"].unique())

    fig, axes = plt.subplots(1, len(METRICAS), figsize=(4 * len(METRICAS), 4))
    for ax, m in zip(axes, METRICAS):
        ax.boxplot([res.loc[res.modelo == k, m] for k in modelos])
        ax.set_xticks(range(1, len(modelos) + 1))
        ax.set_xticklabels(modelos)
        ax.set_title(m)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.suptitle("Métricas por fold y repetición")
    fig.tight_layout()
    fig.savefig(f"{carpeta}/01_metricas.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for k in modelos:
        fpr, tpr, _ = roc_curve(y, oof[k])
        ax.plot(fpr, tpr, label=f"{k} (AUC={roc_auc_score(y, oof[k]):.3f})")
    if p_real is not None:
        fpr, tpr, _ = roc_curve(y, p_real)
        ax.plot(
            fpr, tpr, "k--",
            label=f"Oráculo (AUC={roc_auc_score(y, p_real):.3f})"
        )
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curvas ROC (out-of-fold)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{carpeta}/02_roc.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(modelos), figsize=(4.5 * len(modelos), 4))
    for ax, k in zip(np.atleast_1d(axes), modelos):
        cm = confusion_matrix(y, (oof[k] >= UMBRAL).astype(int))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No alto riesgo", "Alto riesgo"])
        ax.set_yticklabels(["No alto riesgo", "Alto riesgo"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(k)
    fig.tight_layout()
    fig.savefig(f"{carpeta}/03_confusion.png", dpi=130)
    plt.close(fig)

    c = coef_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(c) + 1.5))
    ax.barh(
        c["variable"], c["coef_medio"], xerr=c["coef_std"],
        color=np.where(c["coef_medio"] >= 0, "tab:red", "tab:blue")
    )
    ax.scatter(
        c["beta_asignado"], c["variable"], color="k", marker="D",
        zorder=3, label="beta asignado (generador)"
    )
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Coeficiente (variables estandarizadas)")
    ax.set_title("Regresión logística: coeficientes ± desv. entre folds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{carpeta}/04_coeficientes.png", dpi=130)
    plt.close(fig)

    imp = perm_df.groupby(["variable", "modelo"])["caida_auc"].mean().unstack()
    imp = imp.loc[imp.mean(axis=1).sort_values().index]
    ax = imp.plot.barh(figsize=(9, 0.5 * len(imp) + 2))
    ax.set_xlabel("Caída del AUC al permutar la variable")
    ax.set_title("Importancia por permutación (validación, repetición 1)")
    plt.tight_layout()
    plt.savefig(f"{carpeta}/05_importancia_permutacion.png", dpi=130)
    plt.close()


def guardar_modelos(X, y, carpeta):
    ruta_modelos = os.path.join(carpeta, "modelos")
    os.makedirs(ruta_modelos, exist_ok=True)

    Xv = X.to_numpy(float)
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(Xv)

    for nombre, modelo in construir_modelos(SEMILLA, USAR_KERAS).items():
        print(f"Entrenando modelo final: {nombre}...")
        modelo.fit(X_escalado, y)

        nombre_archivo = (
            nombre.lower()
            .replace(" ", "_")
            .replace("ó", "o")
            .replace("í", "i")
        )

        if isinstance(modelo, MLPKeras):
            modelo.m.save(os.path.join(ruta_modelos, f"{nombre_archivo}.keras"))
            joblib.dump(
                {"escalador": escalador, "columnas": list(X.columns)},
                os.path.join(ruta_modelos, f"{nombre_archivo}_preprocesamiento.joblib")
            )
        else:
            joblib.dump(
                {
                    "modelo": modelo,
                    "escalador": escalador,
                    "columnas": list(X.columns),
                },
                os.path.join(ruta_modelos, f"{nombre_archivo}.joblib")
            )

    print(f"Modelos finales guardados en: {ruta_modelos}/")


def main():
    global USAR_KERAS
    if not os.path.isfile(ARCHIVO_CSV):
        raise FileNotFoundError(
            f"No se encontró '{ARCHIVO_CSV}'. Cambia ARCHIVO_CSV "
            "en la sección CONFIGURACIÓN."
        )

    usar_keras = USAR_KERAS
    if usar_keras:
        try:
            import tensorflow  
        except ImportError:
            print("TensorFlow no está instalado: se usará MLPClassifier de sklearn.")
            usar_keras = False

    USAR_KERAS = usar_keras

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    X, y, p_real = cargar(ARCHIVO_CSV, AGREGAR_INTERACCION)

    print(f"Observaciones: {len(y)} | Variables: {X.shape[1]} -> {list(X.columns)}")
    print(
        f"Clase 1 (alto riesgo): {y.mean() * 100:.1f}% | "
        f"Clase 2: {(1 - y.mean()) * 100:.1f}%"
    )
    print(
        f"Excluidas: {[c for c in EXCLUIR if c != OBJETIVO]} | "
        f"MLP: {'keras' if USAR_KERAS else 'sklearn'} | "
        f"CV: {K_FOLDS}-fold x {REPETICIONES} repeticiones\n"
    )

    res, coefs, perm, oof = experimento(X, y)
    tabla, _ = resumen(res, y, p_real)

    print("=" * 78)
    print(f"RESULTADOS (media ± desv. sobre {K_FOLDS * REPETICIONES} evaluaciones por modelo)")
    print("=" * 78)
    print(tabla.to_string())

    coef_df = tabla_coeficientes(
        coefs, list(X.columns), "escenario" in X.columns
    )
    print("\n" + "=" * 78)
    print("REGRESIÓN LOGÍSTICA: coeficientes (variables estandarizadas)")
    print("=" * 78)
    print(coef_df.round(3).to_string(index=False))

    imp = (
        perm.groupby(["variable", "modelo"])["caida_auc"].mean().unstack()
        .assign(media=lambda d: d.mean(axis=1))
        .sort_values("media", ascending=False)
    )
    print("\n" + "=" * 78)
    print("IMPORTANCIA POR PERMUTACIÓN (caída media del AUC)")
    print("=" * 78)
    print(imp.round(4).to_string())

    res.to_csv(f"{CARPETA_SALIDA}/resultados_por_fold.csv", index=False)
    tabla.to_csv(f"{CARPETA_SALIDA}/resumen_metricas.csv")
    coef_df.to_csv(f"{CARPETA_SALIDA}/coeficientes_lr.csv", index=False)
    imp.to_csv(f"{CARPETA_SALIDA}/importancia_permutacion.csv")
    graficas(res, oof, y, p_real, coef_df, perm, CARPETA_SALIDA)

    guardar_modelos(X, y, CARPETA_SALIDA)

    print(f"\nResultados, gráficas y modelos guardados en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
