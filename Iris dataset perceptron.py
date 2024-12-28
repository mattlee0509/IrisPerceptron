
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


file_path = r'C:\Users\cools\Downloads\iris.csv'

df = pd.read_csv(file_path)


df['Y'] = df['Y'].replace({'setosa': 1, 'versicolor': -1})



margin_mistakes = []
w = np.array([0.0,0.0])


numbers = list(range(0, len(df)))
iterations = 0
min_distance = float('inf')
while True:
    margin_mistake_found = False
    iterations += 1
    
    for index, row in df.iterrows():
        x = np.array(row[['X1', 'X2']])
        y = row['Y']
        if y * np.dot(w, x) < 1:
            margin_mistake_found = True
            random_number = random.choice(numbers)
            x = np.array(df.iloc[random_number, [0, 1]])
            y = df.iloc[random_number, 2]
            if y * np.dot(w, x) < 1:
                w += y * x

    if not margin_mistake_found:
        break
for index, row in df.iterrows():
            x = np.array(row[['X1', 'X2']])
            y = row['Y']
            distance = y * np.dot(w, x)
            if distance < min_distance:
                min_distance = distance
                minx = x
                miny = y
print(min_distance)
print(minx, miny)
print(w)
print(iterations)
slope = -w[0] / w[1]
x_values = np.linspace(min(df['X1']), max(df['X1']), 100)
y_values = slope * x_values

plt.figure(figsize=(8, 6))
plt.scatter(df['X1'], df['X2'], c=df['Y'], cmap='viridis')
plt.plot(x_values, y_values, color='red', label='Decision Boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot of X1 vs X2')
plt.colorbar(label='Y')
plt.show()

