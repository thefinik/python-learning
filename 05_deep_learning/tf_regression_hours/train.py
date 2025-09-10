import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Normalization
from tensorflow.keras.callbacks import EarlyStopping

# 1) Data (hours -> score)
data = pd.DataFrame({
    "hours": [1,2,3,4,5,6,7,8],
    "score": [52,58,63,70,74,79,85,90]
})
X = data[["hours"]].astype("float32").values   # shape: (N, 1)
y = data["score"].astype("float32").values     # shape: (N,)

# 2) Normalization (fit on data)
norm = Normalization()
norm.adapt(X)

# 3) Model: Input → Normalization → Dense(1)
model = Sequential([
    Input(shape=(1,)),
    norm,
    Dense(1, activation="linear")   # 1 neuron, linear output for regression
])

model.compile(optimizer="adam", loss="mse")

# 4) Train (early stopping to avoid overfitting)
es = EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)
history = model.fit(X, y, epochs=2000, verbose=0, callbacks=[es])

# 5) Inspect weights (y ≈ w*x + b after normalization layer)
w, b = model.layers[-1].get_weights()
print("Weight (w):", float(w[0][0]))
print("Bias (b):", float(b[0]))

# 6) Predict a new value (5.5 hours)
pred = model.predict(np.array([[5.5]], dtype="float32"), verbose=0)[0][0]
print("Pred(5.5h):", float(pred))