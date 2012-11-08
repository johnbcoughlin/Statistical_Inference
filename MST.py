from numpy import array, zeros
from numpy.linalg import det

n = 7
adjacency_matrix = zeros(shape=(n, n))
for line in open('../hw4-graph.txt', 'r'):
    v, adjs = line.strip().split(':')
    v = int(v)
    for var in adjs.split():
        var = int(var)
        adjacency_matrix[(v-1, var-1)] = 1
    # Fill the diagonal with the negative sum of the elements in each row
    adjacency_matrix[(v-1, v-1)] = -1 * sum(adjacency_matrix[v-1, :])

laplacian = adjacency_matrix

L_minor = laplacian[0:-1, 0:-1]
print(laplacian)
print(L_minor)

n_MST = det(L_minor)
print(n_MST)

minor_2 = array([[-2, 1, 0, 1],
                    [1, -2, 1, 0],
                    [0, 1, -3, 1],
                    [1, 0, 1, -3]])

print(det(minor_2))
