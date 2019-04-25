import numpy as np 

metadata = np.load('./10_australian_big_metadata.npy')

X = metadata[:, 0:396]
y = metadata[:, 396]


print(np.shape(X))
print(np.shape(y))
print(X[0])
print(y[0:10])