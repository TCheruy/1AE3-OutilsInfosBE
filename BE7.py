# -*- encoding: utf-8 -*-
#  ligne précedente pour pouvoir avoir les accents dans les commentaires

import numpy as np
import matplotlib.pyplot as plt
import time as time

# les deux scripts à compléter en séance:
import relax
import direct


# Choix du schema 1 5pts, 2 9pts
ksch = 1

# méthodes de résolution: 
#=1 -> Jacobi
# =2 -> Gauss-Seidel
# =3 -> SOR
# =4 -> méthode de pivot (factorisation LU générale)
# =5 -> factorisation LU pour matrice creuse
# valeurs prises par la variable ksolve ci-après, enlever les méthodes itératives pour les grands N
# au fur et à mesure ajouter des méthodes dans la liste, exemple:
#solveurs = [1,2,3,4,5]
solveurs=[1,2,3,4,5]
# Valeurs de N à tester
#tailles=[10,20,40]
tailles=[20, 40, 60]

pi = np.pi

#
# résolution
#

# titres de colonnes pour les résultats
print ("ksch | ksolve | N | log10 du pas | log10 de l'erreur | niter | temps cpu (s)")
for ksolve in solveurs:
    niter=0
    for N in tailles:
        # Generation du maillage selon les axes x et y
        #
        h = 1. / (N - 1)  # Pas d'espace dx, dy
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
        
        debtime = time.time()
        if ksolve < 4:  # méthodes itératives
            niter, errtab, restab, T = relax.relax(ksolve, ksch, N, Te)
            err = errtab[niter-1]
            #print ksch,'\t',ksolve,'\t',N,'\t',np.log10(h),'\t', errtab[niter-1],'\t',niter,'\t',time.time()-debtime
        elif ksolve<6:
            err,T =direct.direct(ksolve, ksch, N, Te)
            niter=1
            err = np.log10(err)
            #print ksch,'\t',ksolve,'\t',N,'\t',np.log10(h),'\t', np.log10(err),'\t',niter,'\t',time.time()-debtime
        else: # dernière question uniquement
            import jacobiOptimise as jo
            if ksolve<=7:
                niter, errtab, restab, T = jo.relaxMatrice(ksolve, ksch, N, Te)
            if ksolve==8:
                niter, errtab, restab, T = jo.relaxTableau(ksolve, ksch, N, Te)
            err = errtab[niter-1]
        
        print (ksch,'\t',ksolve,'\t',N,'\t',np.log10(h),'\t', err,'\t',niter,'\t',time.time()-debtime)
	
            

PLOT=True

if PLOT:

    # evolution de la solutions (méthodes itératives)
    X, Y = np.meshgrid(x, y)
    if niter>1 :
            numiter = np.linspace(1, niter-1, niter-1)
            plt.plot(numiter, errtab[0:niter-1], 'r', numiter, restab[0:niter-1], 'b')
            plt.title("erreur en bleu et residu en rouge a chaque iteration")
            plt.xlabel('iterations')
            plt.ylabel('log 10 de l erreur')
            #
            # Graphique contours en 2D
            #
            fig = plt.figure()
            plt.contour(X, Y, T- Te, 10)
            plt.colorbar()
            plt.title(' eccart entre la temperature exacte et celle approchee')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('myfig.png', dpi=300)


    # Graphique en 3D

    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig3d = plt.figure()
    ax = Axes3D(fig3d)
    ax.view_init(30, 300)  #erreur avec python 2.7
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    ax.plot_surface(X,Y,T, rstride=1 if N<=50 else N//50,cstride=1 if N<=50 else N//50,cmap=cm.coolwarm)
    plt.show()
