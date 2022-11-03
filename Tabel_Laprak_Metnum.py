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
