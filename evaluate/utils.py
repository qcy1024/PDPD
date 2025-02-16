import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)
def least_sqaure(N,x_list,y_list,z_list):
    A = np.array([[sum(x_list ** 2), sum(x_list * y_list), sum(x_list)],
                [sum(x_list * y_list), sum(y_list ** 2), sum(y_list)],
                [sum(x_list), sum(y_list), N]])

    B = np.array([[sum(x_list * z_list), sum(y_list * z_list), sum(z_list)]])
    X = np.linalg.solve(A, B.T)
    return X

L = 10
tivecL2 = ti.types.vector(L*L, float)
tivec3 = ti.types.vector(3, float)
solver = ti.linalg.SparseSolver(solver_type='LLT', ordering='AMD')
mat3x3f = ti.types.matrix(3, 3, float)

@ti.func
def pdpd_least_square(N:int, x_list: tivecL2, y_list:tivecL2, z_list:tivecL2 ):
    Asum11 = 0.0
    Asum12 = 0.0
    Asum13 = 0.0
    Asum21 = 0.0
    Asum22 = 0.0
    Asum23 = 0.0
    Asum31 = 0.0
    Asum32 = 0.0
    Asum33 = N * 1.0
    Bsum1 = 0.0
    Bsum2 = 0.0
    Bsum3 = 0.0
    for i in range(L*L):
        Asum11 += x_list[i] ** 2
        Asum12 += x_list[i] * y_list[i]
        Asum13 += x_list[i]
        Asum22 += y_list[i] ** 2
        Asum23 += y_list[i]
        Bsum1 += x_list[i] * z_list[i]
        Bsum2 += y_list[i] * z_list[i]
        Bsum3 += z_list[i]
    Asum32 = Asum23
    Asum31 = Asum13
    Asum21 = Asum12
    A = mat3x3f([[Asum11, Asum12, Asum13], 
         [Asum21, Asum22, Asum23], 
         [Asum31, Asum32, Asum33]])
    B = tivec3([Bsum1, Bsum2, Bsum3])
    A_ivs = A.inverse()
    X = A_ivs @ B
    return X
