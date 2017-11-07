import numpy as np

def portion(mtx, k):

    array = []
    array.append( mtx[:, :k])

    for i in range(1, mtx.shape[1]-1):

        array.append( mtx[:, k*i:k*(i+1)])

    return array[:k+1]

mtx = np.matrix([[1,2,3,10,13,14], [4,5,6,11,15,16], [7,8,9,12,17,18]])
k = 2
print(portion(mtx, k))
print(mtx.shape[1])