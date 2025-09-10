import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Дані
data = pd.DataFrame({
    "hours": [1,2,3,4,5,6,7,8],
    "score": [52,58,63,70,74,79,85,90]
})

# 2) X, y (float32)
X = data[["hours"]].astype("float32").values
y = data["score"].astype("float32").values

# 3) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4) Модель (Заповни: кількість нейронів і активацію)
model = Sequential([
    Input(shape=(1,)),
    Dense(1, activation="linear")  # <- тут впиши кількість нейронів і activation
])

# 5) Компіль і тренування (epochs = 500)
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=500, verbose=0)

# 6) Прогноз і метрики
y_pred = model.predict(X_test, verbose=0).ravel()
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))