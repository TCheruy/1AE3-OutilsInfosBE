
import numpy as np
import matplotlib.pyplot as plot
import time as time

#
# Paramètres
#
ksch = 2  # Choix du schema 1 5pts, 2 9pts
N = 9  # Taille du maillage
print("ksh=",ksch)
print("N= ",N)

h = 1. / (N - 1)  # Pas d'espace
nitermax = 10000  #  nombre maxi d'iteration de jacobi
restab = np.zeros(nitermax)
errtab = np.zeros(nitermax)

pi = np.pi
#
# Generation du maillage selon les axes x et y
#
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
#
# La solution exacte
#
Te = np.zeros([N, N])
for i in range(0, N):
    for j in range(0, N):
        Te[i, j] = (1 / np.sinh(np.pi)) * (
                np.sinh(np.pi * x[i]) * np.sin(np.pi * y[j]) + np.sin(np.pi * x[i]) * np.sinh(np.pi * y[j]))


#   solution approchee  T
T = np.zeros([N, N])

#  Application des conditions aux limites (valeurs imposees en debut de calcul)
T[0, :] = 0.
T[:, 0] = 0.
for i in range(0, N):
    T[N-1, i] = np.sin(pi*y[i])

for i in range(0, N):
    T[i, N-1] = np.sin(pi*x[i])

# Itérations de Jacobi pour converger vers la solution

debtime = time.time() # temps de l'horloge au début du calcul, pour mesure du temps d'exécution
niter = 0  #  compteur nombre d'iterations
res = 1.  #  residu

while res > 1.0E-14 and niter < nitermax:  # on continu les iterations tant que le residu est trop grand
    #  stockage du du niveau N+1 au niveau N
    U = 1*T  # multiplication par "1" pour forcer une copie, sinon U et T deviendraient synonymes
    if ksch == 1:
        #  methode de jacobi a 5 Pts
        for i in range(1, N-1):
            for j in range(1, N-1):
                T[i, j] = 0.25*(U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])

    if ksch == 2:
        #  methode de jacobi a 9 pts
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # a modifier pour passer à 9 pts
                T[i, j] = 0.2 * (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1]) + (1 / 20) * (U[i + 1, j + 1] + U[i - 1, j + 1] + U[i + 1, j - 1] + U[i - 1, j - 1])

    # Calcul du residu en norme L2 - écart entre deux solutions successives
    res = np.linalg.norm(T-U)/N
    restab[niter-1] = np.log10(res)

    # calcul de l'erreur en norme L2 - écart vs. solution exacte
    erreur = np.linalg.norm(T-Te)/N
    errtab[niter - 1] = np.log10(erreur)
    
    niter = niter+1


fintime = time.time() # temps de l'horloge en fin de calcul

print("nombre d'iterations", niter)
print("temps cpu", fintime-debtime)
print("log10 de l'erreur", np.log10(erreur))
print("log10 du pas", np.log10(h))

#
# Graphique
#

numiter = range(niter-1)

fig = plot.figure()
plot.plot(numiter, errtab[0:niter-1], 'r', numiter, restab[0:niter-1], 'b')
plot.title("erreur en rouge et residu en bleu a chaque iteration")
plot.xlabel('iterations')
plot.ylabel('log 10 de l erreur')
plot.grid()
#
# Graphique contours en 2D
#
fig = plot.figure()
X, Y = np.meshgrid(x, y)
plot.contour(X, Y, T- Te, 10)
plot.colorbar()
plot.title(' eccart entre la temperature exacte et celle approchee')
plot.xlabel('x')
plot.ylabel('y')
plot.grid()
#fig.show()
plot.savefig('myfig.png', dpi=300)

##
# Graphique en 3D

from matplotlib import cm

fig3d = plot.figure()


from mpl_toolkits.mplot3d import Axes3D

ax = Axes3D(fig3d)  # erreur avec python 2.7
ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.winter)
plot.title(' temperature dans la plaque')
plot.xlabel('x')
plot.ylabel('y')

plot.show()
