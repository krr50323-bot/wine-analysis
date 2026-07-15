import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/wine.csv')

data_array = df.to_numpy()
X_all = data_array[:, :-1]
y = data_array[:, -1]

x = X_all[:, 0] 

n = len(x)
a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
b = (np.sum(y) - a * np.sum(x)) / n

print(f"Regression Line: y = {a:.4f}x + {b:.4f}")

plt.figure(figsize=(6, 5))

plt.plot(x, y, '.', label='Data points')

x_line = np.linspace(x.min(), x.max(), 100)
y_line = a * x_line + b

plt.plot(x_line, y_line, 'r-', label=f'y = {a:.2f}x + {b:.2f}')

plt.xlabel("Selected Feature") 
plt.ylabel("quality")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

plt.show()
