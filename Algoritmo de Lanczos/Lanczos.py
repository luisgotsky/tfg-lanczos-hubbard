"""
Created on Tuesday 18th of february

@author: Luis Lucas García
"""
import numpy as np
import random
from math import comb
def buildMatrix(d, o, u): #Builds a tridiagonal matrix from its diagonals

    n = len(d)
    M = np.zeros((n, n))
    M[0][0] = d[0]
    
    for i in range(1, n):
        
        M[i][i] = d[i]
        M[i-1][i] = o[i-1]
        M[i][i-1] = u[i-1]
    
    return M

def toBinaryList(num, N):

    b = bin(num)
    B = len(b[2::])
    return [int(b[-1-i]) if i < B else 0 for i in range(N)][::-1]

def toDecList(l):
    
    sum = 0
    L = len(l)
    for i in range(L):
        sum += l[i]*2**(L-i-1)
    return sum

def hamTBbase(vect2): #Hamiltoniano de tight-binding en segunda cuantización, el vector debería de ser de una forma binaria, una lista de 0's y 1's
    
    N = len(vect2)
    newVect = np.zeros(N)
    for i in range(N):
        for j in range(N):
            vect = np.copy(vect2)
            if abs(i - j) == 1:
                if vect[i] == 0 and vect[j] == 1:
                    vect[i] = 1
                    vect[j] = 0
                else: vect = np.zeros(N)
                newVect += vect 
    return newVect

def baseDecomposition(vect, base):
    
    return np.linalg.solve(base, vect)

def inverseBaseDecomposition(coords, base):
    
    N = len(base)
    vect = np.zeros(N)
    for i in range(N):
        vect += coords[i]*base[i]
    return vect

def baseBuilderBinary(N, n): #Construye la base para N átomos con n electrones en una cadena binaria
    if n == 0: return np.zeros(N)
    if n == N: return np.ones(N)
    newBase = np.zeros(N)
    base = []
    for i in range(2**N):
        newBase = toBinaryList(int(toDecList(newBase))+1, N)
        if np.sum(newBase) == n: base.append(newBase)
        if len(base) == N: #Realmente no necesitamos más de N vectores para tener una base, de esta manera le aliviamos algo de estrés al programa
            if np.linalg.det(np.array(base).transpose()) != 0:
                return base
            else: del(base[-1])
    return base

def hamOnBase(H, base): #Devuelve las componentes de H|b> donde b es un vector de la base. Las componentes son en la propia base que se entrega a la función
    coords = []
    for b in base:
        coords.append(baseDecomposition(H(b), base))
    return coords

def hamOnVect(coords, hamBase): #Devuelve las coordenadas de un vector en una base dada tras aplicarle el hamiltoniano
    
    N = len(coords)
    newCoords = np.zeros(N)
    for i in range(N):
        newCoords += coords[i]*hamBase[i]
    return newCoords

def lanczosBinary(H, N, n): #Aplicamos el algoritmo de lanczos en un hamiltoniano de Tight Binding con base dada de forma binaria
    
    base = np.array(baseBuilderBinary(N, n))
    hamBase = hamOnBase(H, base)
    vec0 = base[0] #Tomamos un vector de la base aleatorio como el inicial del algoritmo
    coords0 = baseDecomposition(vec0, base)
    a0 = np.dot(vec0, inverseBaseDecomposition(hamOnVect(coords0, hamBase), base))/np.dot(vec0, vec0)
    b0 = 0
    As, Bs, coords, vecs = [a0], [b0], [coords0], [vec0]
    vec1 = inverseBaseDecomposition(hamOnVect(coords[-1], hamBase), base) - a0*vec0
    coords1 = baseDecomposition(vec1, base)
    vecs.append(vec1)
    coords.append(coords1)
    a1 = np.dot(vec1, inverseBaseDecomposition(hamOnVect(coords1, hamBase), base))/np.dot(vec1, vec1)
    b1 = np.dot(vec1, vec1)/np.dot(vec0, vec0)
    As.append(a1)
    Bs.append(b1)
    
    for i in range(N-2):
        vec = inverseBaseDecomposition(hamOnVect(coords[-1], hamBase), base) - As[-1]*vecs[-1] - Bs[-1]*vecs[-2]
        coordsNew = baseDecomposition(vec, base)
        aNew = np.dot(vec, inverseBaseDecomposition(hamOnVect(coordsNew, hamBase), base))/np.dot(vec, vec)
        bNew = np.dot(vec, vec)/np.dot(vecs[-1], vecs[-1])
        vecs.append(vec)
        coords.append(coordsNew)
        As.append(aNew)
        Bs.append(bNew)
    return buildMatrix(As, np.sqrt(Bs[1::]), np.sqrt(Bs[1::])), vecs, coords, base
    
print(lanczosBinary(hamTBbase, 4, 3))