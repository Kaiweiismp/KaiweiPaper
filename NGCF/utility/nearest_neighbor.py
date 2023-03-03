import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def K_nearest_neighbor(K, u, n):
    number_user = n
    dist_matrix = np.empty([number_user, number_user])
    neighbor_matrix = np.empty([number_user, K])
    
    for i in range(number_user):
        a = u[i]
        for j in range(i, number_user):
            b = u[j]
            temp = cosine(a, b)
            dist_matrix[i][j] = temp
            dist_matrix[j][i] = temp

    for i in range(number_user):
        dist_matrix[i][i] = 0
        
    for i in range(number_user):
        ind = np.argpartition(dist_matrix[i], -K)[-K:]
        neighbor_matrix[i] = ind
    return dist_matrix, neighbor_matrix