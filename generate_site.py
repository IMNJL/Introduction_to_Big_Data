"""Generate a small static site (docs/index.html) showing example inputs,
predicted stress level and simple recommendations.

This script is intentionally lightweight and does not require H2O or cloud APIs.
It trains or loads the scikit-learn demo model created by `simple_demo.py` and
renders a static HTML page into `docs/index.html`.
"""
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_PATH = 'rf_stress_model.joblib'
DATA_PATH = os.path.join('data', 'StressLevelDataset.csv')
OUT_DIR = 'docs'
OUT_FILE = os.path.join(OUT_DIR, 'index.html')


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Train a quick model if not present
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf


def recommend_text(stress_label, sample_row):
    # Simple hand-crafted recommendations based on stress level
    if stress_label == 0:
        return (
            "You're assessed at LOW stress. Keep up healthy routines: maintain good sleep, "
            "short daily breaks, and regular social support."
        )
    if stress_label == 1:
        return (
            "You're assessed at MEDIUM stress. Try 15-minute wind-down routines before bed, "
            "Pomodoro study blocks (25/5), and discuss career concerns with a mentor."
        )
    return (
        "You're assessed at HIGH stress. Please consider reaching out to campus mental health "
        "services or a trusted professional. In the short term, prioritize sleep, reduce study load, "
        "and increase social support."
    )


def html_template(title, sample_inputs, predicted_text, recommendation):
    # Minimal, self-contained HTML page
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{title}</title>
    <style>
      body {{ font-family: Arial, Helvetica, sans-serif; background:#f8fafc; color:#0f172a; padding:24px }}
      .card {{ background: #fff; border-radius:8px; padding:18px; box-shadow: 0 4px 14px rgba(2,6,23,0.06); max-width:900px }}
      h1 {{ margin-top:0 }}
      .muted {{ color:#475569 }}
      .pill {{ background:#ecfeff; padding:8px 12px; border-radius:6px; display:inline-block; margin-top:8px }}
      pre {{ background:#0f172a; color:#f8fafc; padding:12px; border-radius:6px; overflow:auto }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>MindGuard — Live demo (static)</h1>
      <p class="muted">This static page is auto-generated from the repository demo model. It shows a sample input, the predicted stress level, and simple recommendations.</p>

      <h3>Sample input</h3>
      <pre>{sample_inputs}</pre>

      <h3>Predicted stress level</h3>
      <div class="pill">{predicted_text}</div>

      <h3 style="margin-top:18px">Recommendation</h3>
      <p>{recommendation}</p>
    </div>
  </body>
</html>
"""


def build_site():
    os.makedirs(OUT_DIR, exist_ok=True)
    clf = ensure_model()
    df = pd.read_csv(DATA_PATH)
    sample = df.drop(columns=['stress_level']).iloc[0:1]
    pred = clf.predict(sample)[0]
    mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    predicted_text = f"{pred} ({mapping.get(pred, 'Unknown')})"
    rec = recommend_text(pred, sample.iloc[0].to_dict())
    sample_json = sample.to_dict(orient='records')[0]
    html = html_template('MindGuard Demo', sample_json, predicted_text, rec)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Wrote static site to {OUT_FILE}")


if __name__ == '__main__':
    build_site()
