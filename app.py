from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and clean the dataset
def load_data(remove_outliers=True):
    df = pd.read_csv("data_cleaned.csv")
    df.columns = df.columns.str.strip()

    df = df[df['year'].between(1990, 2000)]
    df.drop(columns=['country'], inplace=True, errors='ignore')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['co2_per_cap'].notna()]
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("All rows removed after cleaning.")

    if remove_outliers:
        numeric_df = df.select_dtypes(include=np.number)
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        df = df.loc[mask]
        if df.empty:
            raise ValueError("All rows removed by outlier filtering.")

    return df

@app.route("/", methods=["GET", "POST"])
def index():
    metrics = None
    prediction = None
    error = None
    graph_url = None

    try:
        df = load_data(remove_outliers=False)

        # === Generate CO₂ Emissions Graph ===
        plot_df = df.groupby("year")["co2_per_cap"].mean().reset_index()

        plt.figure(figsize=(6, 4))
        plt.plot(plot_df["year"], plot_df["co2_per_cap"], marker='o')
        plt.xlabel("Year")
        plt.ylabel("Avg CO₂ per Capita")
        plt.title("CO₂ Emissions Over Time (1990–2000)")
        plt.grid(True)

        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.read()).decode()
        plt.close()

        # === Handle POST: Train or Predict ===
        if request.method == "POST":
            if 'train' in request.form:
                X = df.drop(columns=['co2_per_cap'])
                y = df['co2_per_cap']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(random_state=42)
                }

                results = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    results[name] = {
                        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                        "MSE": round(mean_squared_error(y_test, y_pred), 4),
                        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                        "R2": round(r2_score(y_test, y_pred), 4)
                    }

                metrics = results

            elif 'predict' in request.form:
                user_input = {
                    'year': float(request.form['year']),
                    'gdp': float(request.form['gdp']),
                    'pop': float(request.form['pop']),
                    'en_per_cap': float(request.form['en_per_cap']),
                    'fdi_perc_gdp': float(request.form['fdi_perc_gdp'])
                }

                X = df.drop(columns=['co2_per_cap'])
                y = df['co2_per_cap']

                for col in X.columns:
                    if col not in user_input:
                        user_input[col] = 0.0

                input_df = pd.DataFrame([user_input])[X.columns]

                model = RandomForestRegressor(random_state=42)
                model.fit(X, y)

                scaler = MinMaxScaler()
                scaler.fit(X)
                input_scaled = scaler.transform(input_df)

                prediction = round(model.predict(input_scaled)[0], 4)

    except Exception as e:
        error = str(e)

    return render_template("index.html", metrics=metrics, prediction=prediction, error=error, graph_url=graph_url)

if __name__ == "__main__":
    app.run(debug=True,port=5002)










