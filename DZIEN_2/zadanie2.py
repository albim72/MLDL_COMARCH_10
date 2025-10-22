#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Iris Random Forest – pełny pipeline:
- przygotowanie danych (DataFrame),
- podział train/test (70/30),
- trenowanie RandomForestClassifier,
- ocena: accuracy, classification report, confusion matrix, feature importances,
- wizualizacje: macierz pomyłek (kolorowa) oraz 2D scatter (2 cechy, kolory = klasy przewidziane).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


@dataclass
class IrisData:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    target_names: list[str]


def load_iris_as_df() -> IrisData:
    """Wczytaj Iris jako DataFrame i Series."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    return IrisData(X=X, y=y, feature_names=feature_names, target_names=target_names)


def split_data(
    iris: IrisData, test_size: float = 0.30, random_state: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Podział danych na train/test."""
    X_train, X_test, y_train, y_test = train_test_split(
        iris.X, iris.y, test_size=test_size, random_state=random_state, stratify=iris.y
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 1
) -> RandomForestClassifier:
    """Trenowanie modelu RandomForestClassifier."""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def evaluate_and_print_metrics(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str],
) -> dict:
    """Policz metryki i wypisz na konsolę. Zwróć je też w dict."""
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Wyniki klasyfikacji ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (y_true x y_pred):")
    print(cm)

    # Ważność cech
    importances = clf.feature_importances_
    feat_importances = pd.Series(importances, index=X_test.columns).sort_values(ascending=False)
    print("\nFeature importances:")
    for name, val in feat_importances.items():
        print(f"- {name}: {val:.4f}")

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "feature_importances": feat_importances,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    cm: np.ndarray, target_names: list[str], save_path: str = "confusion_matrix.png"
) -> None:
    """Rysuj kolorową macierz pomyłek i zapisz do pliku."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    # Oś i etykiety
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        ylabel="Prawdziwa klasa",
        xlabel="Przewidziana klasa",
        title="Macierz pomyłek (Iris — Random Forest)",
    )

    # Wypisz wartości w komórkach
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_2d_scatter_predictions(
    X: pd.DataFrame,
    y_pred: np.ndarray,
    feature_names: list[str],
    save_path: str = "scatter_2d_predictions.png",
    feat_idx_a: int = 0,
    feat_idx_b: int = 1,
) -> None:
    """
    Rysuj 2D scatter dwóch cech, kolorując punkty wg klasy PRZEWIDZIANEJ przez model.
    Domyślnie używa dwóch pierwszych cech zbioru.
    """
    feat_a = feature_names[feat_idx_a]
    feat_b = feature_names[feat_idx_b]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    scatter = ax.scatter(
        X.iloc[:, feat_idx_a],
        X.iloc[:, feat_idx_b],
        c=y_pred,  # kolory odpowiadają klasom przewidzianym
        alpha=0.85,
        edgecolors="none",
    )

    ax.set_xlabel(feat_a)
    ax.set_ylabel(feat_b)
    ax.set_title("Iris — przewidziane klasy (2D, dwie pierwsze cechy)")

    # Prosta legenda: tworzymy 3 proxy-handles dla klas 0/1/2
    classes = np.unique(y_pred)
    handles = []
    labels = []
    cmap = scatter.cmap
    norm = scatter.norm
    for c in classes:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(norm(c))))
        labels.append(str(c))
    ax.legend(handles, labels, title="Klasa (pred.)", loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest na Iris – pełny pipeline")
    parser.add_argument("--n_estimators", type=int, default=100, help="Liczba drzew w lesie (domyślnie 100)")
    parser.add_argument("--random_state", type=int, default=1, help="Ziarno losowości (domyślnie 1)")
    parser.add_argument("--test_size", type=float, default=0.30, help="Udział zbioru testowego (domyślnie 0.30)")
    args = parser.parse_args()

    # 1) Wczytanie danych
    iris = load_iris_as_df()

    # 2) Podział danych
    X_train, X_test, y_train, y_test = split_data(
        iris, test_size=args.test_size, random_state=args.random_state
    )

    # 3) Trenowanie modelu
    clf = train_random_forest(
        X_train, y_train, n_estimators=args.n_estimators, random_state=args.random_state
    )

    # 4) Ocena i metryki
    results = evaluate_and_print_metrics(clf, X_test, y_test, iris.target_names)

    # 5) Wizualizacje
    plot_confusion_matrix(results["confusion_matrix"], iris.target_names, "confusion_matrix.png")
    # scatter 2D na danych testowych – kolory = klasy przewidziane przez model
    plot_2d_scatter_predictions(
        X_test.reset_index(drop=True),
        results["y_pred"],
        iris.feature_names,
        "scatter_2d_predictions.png",
        feat_idx_a=0,
        feat_idx_b=1,
    )

    # 6) Dodatkowo: zapis rankingów ważności do CSV (opcjonalnie)
    results["feature_importances"].to_csv("feature_importances.csv", header=["importance"])

    print("\nPliki zapisane:")
    print("- confusion_matrix.png")
    print("- scatter_2d_predictions.png")
    print("- feature_importances.csv")


if __name__ == "__main__":
    main()
