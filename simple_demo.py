"""Simple demo: train a RandomForest on the sample dataset and save a model.

Run this script after installing requirements. It trains a small classifier and saves
`rf_stress_model.joblib` in the repository root.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os


DATA_PATH = os.path.join('data', 'StressLevelDataset.csv')


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def train_and_save(df, model_path='rf_stress_model.joblib'):
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, preds))

    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def demo_predict(model_path):
    clf = joblib.load(model_path)
    df = load_data()
    sample = df.drop(columns=['stress_level']).iloc[0:1]
    print("\nSample input:\n", sample.to_dict(orient='records')[0])
    pred = clf.predict(sample)
    mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    print(f"Predicted stress level for sample: {pred[0]} ({mapping.get(pred[0], 'Unknown')})")


if __name__ == '__main__':
    df = load_data()
    model_file = train_and_save(df)
    demo_predict(model_file)
