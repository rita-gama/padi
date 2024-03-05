import numpy as np

P = np.array([[0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]
])

print(P)
print(P.shape)

c = np.array([[0], 
              [1],
              [0],
              [1],
              [1],
              [1],
              [1],
])

print(c)
print(c.shape)

I = np.eye(N=7, M=7)
print(I)
print(I.shape)

gamma = 0.9


aux = I - (np.dot(gamma, P))
print(aux)
J = np.dot(np.linalg.inv(aux), c)
print(J)