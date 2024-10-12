import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def plot_decision_boundaries(X, y, model, save_path=None):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.coolwarm)

    handles, labels = scatter.legend_elements()
    class_labels = ['Setosa', 'Versicolor']
    plt.legend(handles, class_labels, title="Especies")

    plt.xlabel('Longitud del Sepal (cm)')
    plt.ylabel('Anchura del Sepal (cm)')
    plt.title('Clasificación de Flores Iris')

    if save_path:
        plt.savefig(save_path, format='png', dpi=500)

    plt.show()


def main():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    df = df[df['target'] != 2]
    X = df[['sepal length (cm)', 'sepal width (cm)']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    svm_model = SVC(kernel='rbf', gamma='auto')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precision del modelo SVM: {accuracy * 100:.2f}%')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"svm_decision_boundary_{timestamp}.png")

    plot_decision_boundaries(X_test, y_test, svm_model, save_path)


if __name__ == "__main__":
    main()
