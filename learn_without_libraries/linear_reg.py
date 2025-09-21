import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
X = np.linspace(1, 10, 30)  
y = 3 * X + 7 + np.random.randn(30) * 2  

df = pd.DataFrame({"X": X, "y": y})

# def loss_function(m, b, points):
#     total_error = 0
#     for i in range(len(points)):
#         x = points.iloc[i].X
#         y = points.iloc[i].y
#         total_error += (y - (m * x + b))** 2
#     total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].y

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, df, L)

print(m, b)

plt.scatter(df.X, df.y, color="blue")

x_start = int(df.X.min() )
x_end = int(df.X.max() )

plt.plot(list(range(x_start, x_end)), [m * x + b for x in range(x_start, x_end)], color = "red")
plt.show()