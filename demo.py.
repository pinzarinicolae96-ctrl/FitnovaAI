demo.py.

# demo.py
# Simple AI demo model for FitnovaAI
# Author: Aurel Pinzari
# Date: 2025

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generate some dummy training data
# X = features (e.g. steps, calories, sleep hours)
# y = labels (0 = needs rest, 1 = can train)
X = np.random.rand(100, 3)
y = np.random.randint(2, size=(100, 1))

# Define a simple neural network
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training model...")
history = model.fit(X, y, epochs=10, verbose=1)

# Evaluate the model
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Training Accuracy: {acc:.2f}")

# Make a dummy prediction
sample = np.array([[0.7, 0.2, 0.8]])  # steps=0.7, calories=0.2, sleep=0.8
prediction = model.predict(sample)
print(f"Sample prediction (0=rest, 1=train): {prediction[0][0]:.2f}")
