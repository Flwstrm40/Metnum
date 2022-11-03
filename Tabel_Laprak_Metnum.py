# ----------------------------------------------------------------------------- EULER ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
%matplotlib inline

# Definisi parameter
f = lambda t, s: t  + s# ODE

# Metode Euler
def euler(f,x0,y0,xn,n):
    h = (xn-x0)/n # ukuran langkah
    print("h = ",h)
    t = np.arange(x0, xn + h, h) # grid Numerik
    s0 = y0 # kondisi awal
    
    # Metode Euler
    s = np.zeros(len(t))
    s[0] = s0
    
    for i in range(0, len(t) - 1):
        s[i + 1] = s[i] + h*f(t[i], s[i])
        t[i + 1] = t[i] + h
        
    return t,s
print('Masukkan kondisi awal :')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

print('Masukkan titik yang dihitung : ')
xn = float(input('xn = '))

print('Masukkan jumlah step :')
n = int(input('Jumlah step = '))

# menampilkan grafik
t,s = euler(f,x0,y0,xn,n)
plt.figure(figsize = (12, 8))
plt.plot(t, s, 'bo--', label='Aproksimasi')
plt.plot(t, np.exp(t) - t - 1, 'g', label='Exact')
plt.title('Solusi Aproksimasi dan Exact untuk ODE sederhana')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# perhitungan galat
sol_euler = s[n]
sol_exact = np.exp(0.1) - 0.1 - 1
galat_ra_euler = abs((sol_exact-sol_euler)/sol_exact)*100
print('Solusi Euler =',sol_euler)
print('Solusi Exact =',sol_exact)
print('galat relatif absolut = ',galat_ra_euler,'%')

# ----------------------------------------------------------------------------- Heun ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
%matplotlib inline

# Definisi parameter
f = lambda t, s: t + s # ODE

# Metode Heun
def heun(f,x0,y0,xn,n):
    h = (xn-x0)/n # ukuran langkah
    print("h = ",h)
    t = np.arange(x0,xn+h,h) # grid Numerik
    s0 = y0 # kondisi awal
    
    # Metode heun
    s = np.zeros(len(t))
    s[0] = s0
    
    for i in range(0, len(t) - 1):
        k1 = h*f(t[i], s[i])
        k2 = h*f(t[i]+h, s[i]+k1)
        s[i + 1] = s[i] + 0.5*(k1+k2)
        t[i+1] = t[i]+h
    return t, s

print('Masukkan kondisi awal :')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

print('Masukkan titik yang dihitung : ')
xn = float(input('xn = '))

print('Masukkan jumlah step :')
n = int(input('Jumlah step = '))

# menampilkan grafik
t,s = heun(f,x0,y0,xn,n)
plt.figure(figsize = (12, 8))
plt.plot(t, s, 'bo--', label='Aproksimasi')
plt.plot(t, np.exp(t) - t - 1, 'g', label='Exact')
plt.title('Solusi Aproksimasi dan Exact untuk ODE sederhana')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# perhitungan galat
sol_heun = s[n]
sol_exact = np.exp(0.1) - 0.1 - 1
galat_ra_heun = abs((sol_exact-sol_heun)/sol_exact)*100
print('ya =',sol_heun)
print('ys =',sol_exact)
print('galat relatif absolut = ',galat_ra_heun,'%')

# ----------------------------------------------------------------------------- RK4 ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
%matplotlib inline

# Definisi parameter
f = lambda t, s: t + s # ODE

# Metode Runge Kutta orde 4
def rungekutta4(f,x0,y0,xn,n):
    h = (xn-x0)/n # ukuran langkah
    print("h = ",h)
    t = np.arange(x0,xn+h,h) # grid Numerik
    s0 = y0 # kondisi awal
    
    # Metode Runge Kutta orde 4
    s = np.zeros(len(t))
    s[0] = s0
    
    for i in range(0, len(t) - 1):
        k1 = h*f(t[i], s[i])
        k2 = h*f(t[i]+ 0.5*h, s[i]+ 0.5*k1)
        k3 = h*f(t[i]+ 0.5*h, s[i]+ 0.5*k2)
        k4 = h*f(t[i]+ h, s[i]+k3)
        s[i + 1] = s[i] + (1/6)*(k1 + 2*k2 + 2*k3+ k4)
        t[i+1] = t[i]+h
    return t, s

print('Masukkan kondisi awal :')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

print('Masukkan titik yang dihitung : ')
xn = float(input('xn = '))

print('Masukkan jumlah step :')
n = int(input('Jumlah step = '))

# menampilkan grafik
t,s = rungekutta4(f,x0,y0,xn,n)
plt.figure(figsize = (12, 8))
plt.plot(t, s, 'bo--', label='Aproksimasi')
plt.plot(t, np.exp(t) - t - 1, 'g', label='Exact')
plt.title('Solusi Aproksimasi dan Exact untuk ODE sederhana')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# perhitungan galat
sol_rk = s[n]
sol_exact = np.exp(0.1) - 0.1 - 1
galat_ra_rk = abs((sol_exact-sol_rk)/sol_exact)*100
print("Solusi RungeKutta Orde 4 = ",sol_rk)
print("Solusi Exact = ",sol_exact)
print('galat relatif absolut = ',galat_ra_rk,'%')


# ----------------------------------------------------------------------------- TABEL ---------------------------------------------------------------------------

print("*******************************************************************************************************************************")
print("\t\tProgram untuk Menyelesaian Persamaan Differensial")
print("\t\t\tdy/dx = x+y ; syarat y(0)=1")
print("\t\tdengan Metode Euler, Heun, dan Runge Kutta4")
print("\t\tDibuat oleh :")
print("\t\t\tNIM         :\t24060121130073")
print("\t\t\tProg. Studi :\tInformatika")
print("*******************************************************************************************************************************")
print("\t\t\t\t\tSolusi PDB dy/dx= 𝑥 + 𝑦, 𝑦(0) = 1.")

# Tabel
h = 0.01
from texttable import Texttable
l = [["x", "h", "y_Analitik", "y_Euler", "Error %", "y_Heun", "Error %", "y_RungeKutta4", "Error %"]]
m = [[xn, h, sol_exact, sol_euler, galat_ra_euler, sol_heun, galat_ra_heun, sol_rk, galat_ra_rk]]

table = Texttable()
table.set_cols_width([5,5,12.5,12.5,12.5,12.5,12.5,12.5,12.5])
table.add_rows(l)
print(table.draw())
table.add_rows(m)
print(table.draw())
