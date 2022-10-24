## module gaussSeidel
""" x = gaussSeidel(a,b).
Menyelesaikan [a]{b} = {x} dengan Eliminasi Gauss Seidel .
"""
import numpy as np
def gaussSeidel(a,b):
    n = len(b)
    x = np.zeros(n)
    tol = 0.0001
    temp = 0
    
    xold  = np.copy(x)
    for i in range (n):
        for j in range (n):
            if j != i:
                temp += a[i,j]*x[j]
        x[i] = (1/a[i,i])*(b[i]-temp)
        temp = 0
        
    print("x = ", x)
    print("xold = ", xold)
    while (abs(x[0]-xold[0]) > tol ):
        print("error =", abs(x-xold))
        print("\n")
        xold = x.copy()
        for i in range (n):
            for j in range (n):
                if j != i:
                    temp += a[i,j]*x[j]
            x[i] = (1/a[i,i])*(b[i]-temp)
            temp = 0
        print("x = ", x)
        print("xold = ", xold)
        
    print("\n\nhasil akhir x = ",x)
    
# soal
a = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]])
b = np.array([7.85, -19.3, 71.4])

print("Persoalan")
print("a = \n", a)
print("b = \n", b)
