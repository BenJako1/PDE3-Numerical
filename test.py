import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the equation
def equation(x, t, k, n_max=50):
    result = (-x * t) / (0.5 * np.pi)
    for n in range(1, n_max + 1):
        result += (np.power(-1, n) / np.power(n, 3)) * (1 - np.exp(-4 * np.power(n, 2) * t))
    result += np.sin(x) * np.exp(-k * t) * (0.5 / np.pi)
    return result

# Generate data points
x = np.linspace(-5, 5, 100)
t = np.linspace(0, 5, 100)
x, t = np.meshgrid(x, t)
k = 0.2  # Updated value of k

# Calculate the function values
z = equation(x, t, k)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, z, cmap='viridis')

# Labels
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('z')

# Title
ax.set_title('3D Plot of the Equation (k = 0.2, t from 0 to 5)')

plt.show()