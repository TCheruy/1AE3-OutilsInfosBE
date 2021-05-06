# -*- encoding: utf-8 -*-
#  ligne précedente pour pouvoir avoir les accents dans les commentaires

import numpy as np


def relax(ksolve, ksch, N, Te):  #

    niter=0  # nombre d'itération des méthodes itératives
    nitermax = 20000  # nombre maxi d'iteration de jacobi
    restab = np.zeros(nitermax)
    errtab = np.zeros(nitermax)
    pi = np.pi
    #
    # Generation du maillage selon les axes x et y
    #
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
#
#    solution approchee  T
    T = np.zeros([N, N])

    nitermax = 20000  #  nombre maxi d'iteration de jacobi
#
    om = 2. * (1 - np.pi/N)
#
#  Application des conditions aux limites (valeurs imposees en debut de calcul)
    T[0, :] = 0.
    T[:, 0] = 0.
    for i in range(0, N):
         T[N-1, i] = np.sin(pi*y[i])

    for i in range(0, N):
        T[i, N-1] = np.sin(pi*x[i])

    res = 1 #résidu
    while res > 1.0E-8 and niter < nitermax:  # on continu les iterations tant que le residu est trop grand
        #  stockage du du niveau N+1 au niveau N
        U = T.copy()
        if ksolve == 1:  # méthode de jacobi	    
            if ksch == 1:
                #  methode de jacobi a 5 Pts
                for i in range(1, N-1):
                    for j in range(1, N-1):
                        T[i, j] = 0.25*(U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])

            if ksch == 2:
                #  methode de jacobi a 9 pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j]= 0.2*(U[i+1,j]+U[i-1,j]+U[i,j+1]+U[i,j-1])+ 0.05*(U[i+1,j+1]+U[i-1,j-1]+U[i+1,j-1]+U[i-1,j+1])

        if ksolve == 2:  # méthode de Gauss Seidel
            if ksch == 1:
                ##  methode GS a 5 Pts
                for i in range(1, N-1):
                    for j in range(1, N-1):
                        T[i, j] = 0.25*(T[i-1, j] + U[i+1, j] + T[i, j-1]+U[i, j+1])
                

            if ksch == 2:
                ##  methode GS a 9 pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = 0.2 * (U[i + 1, j] + T[i - 1, j] + U[i, j + 1] + T[i, j - 1]) + 0.05 * ( U[i + 1, j + 1] + T[i - 1, j - 1] + U[i + 1, j - 1] + U[i - 1, j + 1])

        if ksolve == 3:  # méthode SOR
            if ksch == 1:
                #  methode SOR a 5 Pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = (1-om)*U[i, j] + om*(0.25*(T[i-1, j]+U[i+1, j]+T[i, j-1]+U[i, j+1]))

            if ksch == 2:
                #  methode SOR a 9 pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = (1-om)*U[i, j] + om*(0.2 * (U[i + 1, j] + T[i - 1, j] + U[i, j + 1] + T[i, j - 1]) + 0.05 * (U[i + 1, j + 1] + T[i - 1, j - 1] + U[i + 1, j - 1] + U[i - 1, j + 1]))
                

        niter = niter+1
       
        # Calcul du residu en norme L2
        res = np.linalg.norm(T-U)/N
        restab[niter-1] = np.log10(res)

        # calcul de l'erreur en norme L2
        err = np.linalg.norm(T-Te)/N
        errtab[niter-1] = np.log10(err)
        
        

    return (niter, errtab, restab, T)