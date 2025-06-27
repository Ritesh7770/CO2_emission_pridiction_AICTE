

# 1. Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import PredictionErrorDisplay

# 2. Load Dataset
data = pd.read_csv("C:\\Users\\HP PC\\Desktop\\coding\\java\\co2_predictiion\\co2_prediction.csv")

print("Data Loaded:", data.shape)

# 3. Basic Cleaning
data = data.dropna()
print("After dropping missing values:", data.shape)

# 4. Encode Categorical Columns
if 'country' in data.columns:
    le = LabelEncoder()
    data['country'] = le.fit_transform(data['country'])

# 5. Feature Selection
features = ['gdp', 'energy_per_cap', 'pop', 'methane', 'nitrous', 'country', 'year']
target = 'co2_per_cap'
X = data[features]
y = data[target]

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predict and Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Model RMSE: {rmse:.2f}")
print(f"Model R² Score: {r2:.2f}")


disp = PredictionErrorDisplay.from_predictions(
    y_true=y_test.values,
    y_pred=y_pred,
    kind="actual_vs_predicted",
    line_kwargs={"color":"red","linewidth":2},
    scatter_kwargs={"alpha":0.6, "edgecolors":'w'}
)
disp.plot()
plt.title("Actual vs Predicted (sklearn PredictionErrorDisplay)")
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', label='Predicted vs Actual')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2, label='Ideal (y = x)')
plt.xlabel("Actual CO₂ per Capita")
plt.ylabel("Predicted CO₂ per Capita")
plt.title("📊 Predicted vs Actual CO₂ per Capita")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()