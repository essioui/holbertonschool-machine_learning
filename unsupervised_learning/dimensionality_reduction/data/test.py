import numpy as np

vector = np.ones((100, 1))

m1 = vector[55]
m2 = vector[55, 0]

vector[55] = 2  

print(m1)
print(m2)
