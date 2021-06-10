import numpy as np

A = np.array(
    [
        [0, 2, 4],
        [2, 4, 2],
        [3, 3, 1]
    ]
)
A_inverse = np.linalg.inv(A)

print('Problem A.11.a')
print(A_inverse)

b = np.array(
    [
        [-2], [-2], [-4]
    ]
)
c = np.array(
    [
        [1], [1], [1]
    ]
)

print('Problem A.11.b')
print(np.dot(A_inverse, b))
print(np.dot(A, c))
