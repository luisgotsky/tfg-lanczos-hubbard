"""
Creado el lunes 1 de abril de 2025

@author: Luis Lucas García
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

e0 = 0
t = 0.5
U = 8

class Vector:
    
    def __init__(self, coords, wordsUp, wordsDown, N):
        cCoords, cUp, cDown = cleanup(coords, wordsUp, wordsDown)
        self.coords, self.up, self.down, self.N = cCoords, cUp, cDown, N
        
    def prod(self, vect):
        coordsA, upA, downA = self.coords, self.up, self.down
        coordsB, upB, downB = vect.coords, vect.up, vect.down
        sum = 0
        for i in range(len(coordsA)):
            for j in range(len(coordsB)):
                if upA[i] == upB[j] and downA[i] == downB[j]:
                    c = coordsA[i]*coordsB[j]
                    sum += c
        return sum
    
    def norm(self):
        if self.up == [0]*len(self.coords) and self.down == [0]*len(self.coords): return 1
        return self.prod(self)
    
    def __add__(self, b):
        newCoords = self.coords + b.coords
        newUp = self.up + b.up
        newDown = self.down + b.down
        return Vector(newCoords, newUp, newDown, self.N)
    
    def __sub__(self, b):
        newCoords = [-b.coords[i] for i in range(len(b.coords))]
        return self + Vector(newCoords, b.up, b.down, self.N)
    
    def __mul__(self, c):
        newCoords = [c*self.coords[i] for i in range(len(self.coords))]
        return Vector(newCoords, self.up, self.down, self.N)
    
    def __str__(self):
        string = str(self.coords) + " " + str(self.up) + " " + str(self.down)
        return string
        

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

def cleanup(coords, upWords, downWords):

    cleanCoords, cleanUp, cleanDown = [], [], []
    for i in range(len(coords)):
        if coords[i] != 0:
            isIn = False
            for j in range(len(cleanCoords)):
                if upWords[i] == cleanUp[j] and downWords[i] == cleanDown[j]:
                    cleanCoords[j] += coords[i]
                    isIn = True
                    break
            if not isIn:
                cleanCoords.append(coords[i])
                cleanUp.append(upWords[i])
                cleanDown.append(downWords[i])
    return cleanCoords, cleanUp, cleanDown

def hubbardOnWord(N, wordUp, wordDown, tMat, Ulist): #Hamiltoniano de Hubbard actuando en un par de palabras spin up-spin down. Dado que wordUp y wordDown son nÃºmeros necesitamos la string.
    
    wordUp = toBinaryList(wordUp, N)
    wordDown = toBinaryList(wordDown, N)
    upWords, downWords, coef = [], [], []
    mutUp = wordUp.copy()
    mutDown = wordDown.copy()
    for i in range(N):
        if wordUp[i] == 1 and wordDown[i] == 1:
            upWords.append(toDecList(wordUp))
            downWords.append(toDecList(wordDown))
            coef.append(Ulist[i] + 2*tMat[i][i])
        elif wordUp[i] == 1 or wordDown[i] == 1:
            upWords.append(toDecList(wordUp))
            downWords.append(toDecList(wordDown))
            coef.append(tMat[i][i])
        for j in range(N):
            if wordUp[i] == 0 and wordUp[j] == 1:
                coef.append(tMat[i][j])
                mutUp[j] = 0
                mutUp[i] = 1
                upWords.append(toDecList(mutUp))
                downWords.append(toDecList(mutDown))
                mutUp = wordUp.copy()
            if wordDown[i] == 0 and wordDown[j] == 1:
                coef.append(tMat[i][j])
                mutDown[j] = 0
                mutDown[i] = 1
                downWords.append(toDecList(mutDown))
                upWords.append(toDecList(mutUp))
                mutDown = wordDown.copy()
    return coef, upWords, downWords

def hubbardOnVec(vec):
    """
    coords, upWords, downWords = vec.coords, vec.up, vec.down
    newUp, newDown, newCoords = [], [], []
    for i in range(len(coords)):
        coord, up, down = hubbardOnWord(N, upWords[i], downWords[i], tMat, Ulist)
        newCoords += [coords[i]*coord[j] for j in range(len(coord))]
        newUp += up
        newDown += down
    """
    newVec = Vector([0], [0], [0], N)
    for i in range(N):
        for j in range(N):
            newVec += creationVec(j, destructionVec(i, vec, "up", N), "up", N)*tMat[i][j] + creationVec(j, destructionVec(i, vec, "down", N), "down", N)*tMat[i][j]
        newVec += creationVec(i, creationVec(i, destructionVec(i, destructionVec(i, vec, "up", N), "down", N), "down", N), "up", N)*Ulist[i]
    return newVec

def lanczos(vec0, dimn):
    hubOnLast = hubbardOnVec(vec0)
    #Pasos iniciales algoritmo de Lanczos
    bSq = [0]
    aS = [vec0.prod(hubOnLast)/vec0.norm()]
    vec1 = hubbardOnVec(vec0) - vec0*aS[0]
    vecs = [vec0, vec1]
    #Algoritmo de Lanczos
    for i in range(dimn-2):
        lastVec = vecs[-1]
        hubOnLast = hubbardOnVec(lastVec)
        aS.append(lastVec.prod(hubOnLast)/lastVec.norm())
        bSq.append(lastVec.norm()/vecs[-2].norm())
        vecs.append(hubOnLast - lastVec*aS[-1] - vecs[-2]*bSq[-1])
    #Paso final algoritmo de Lanczos
    lastVec = vecs[-1]
    aS.append(lastVec.prod(hubbardOnVec(lastVec))/lastVec.norm())
    bSq.append(lastVec.norm()/vecs[-2].norm())
    return aS, bSq, vecs

def groundState(coords, vecs):
    
    newVec = Vector([0], [0], [0], vecs[0].N)
    for i in range(len(coords)):
        newVec += vecs[i]*coords[i]
    return newVec

def nParticles(i, vec, s, N):
    return vec.prod(creationVec(i, destructionVec(i, vec, s, N), s, N))
    
def nHoles(i, vec, s, N):
    return vec.prod(destructionVec(i, creationVec(i, vec, s, N), s, N))
    
def nnParticles(i, j, s1, s2, vec, N):

    return vec.prod(creationVec(i, creationVec(j, destructionVec(i, destructionVec(j, vec, s1, N), s2, N), s1, N), s2, N))

def nnHoles(i, j, s1, s2, vec, N):

    return vec.prod(destructionVec(i, destructionVec(j, creationVec(i, creationVec(j, vec, s1, N), s2, N), s1, N), s2, N))

def nPnH(i, j, s1, s2, vec, N):

    return vec.prod(creationVec(i, destructionVec(j, destructionVec(i, creationVec(j, vec, s1, N), s2, N), s1, N), s2, N))

def nHnP(i, j, s1, s2, vec, N):

    return vec.prod(destructionVec(i, creationVec(j, creationVec(i, destructionVec(j, vec, s1, N), s2, N), s1, N), s2, N))
    
def creationVec(i, vec, s, N):
    newCoords, newUp, newDown = [], [], []
    if s == "up":
        for j in range(len(vec.coords)):
            mutUp = toBinaryList(vec.up[j], N)
            if mutUp[i] == 0:
                mutUp[i] = 1
                newUp.append(toDecList(mutUp))
                newDown.append(vec.down[j])
                newCoords.append(vec.coords[j])
        return Vector(newCoords, newUp, newDown, N)
    if s == "down":
        for j in range(len(vec.coords)):
            mutDown = toBinaryList(vec.down[j], N)
            if mutDown[i] == 0:
                mutDown[i] = 1
                newUp.append(vec.up[j])
                newDown.append(toDecList(mutDown))
                newCoords.append(vec.coords[j])
        return Vector(newCoords, newUp, newDown, N)
    
def destructionVec(i, vec, s, N):
    newCoords, newUp, newDown = [], [], []
    if s == "up":
        for j in range(len(vec.coords)):
            mutUp = toBinaryList(vec.up[j], N)
            if mutUp[i] == 1:
                mutUp[i] = 0
                newUp.append(toDecList(mutUp))
                newDown.append(vec.down[j])
                newCoords.append(vec.coords[j])
        return Vector(newCoords, newUp, newDown, N)
    if s == "down":
        for j in range(len(vec.coords)):
            mutDown = toBinaryList(vec.down[j], N)
            if mutDown[i] == 1:
                mutDown[i] = 0
                newUp.append(vec.up[j])
                newDown.append(toDecList(mutDown))
                newCoords.append(vec.coords[j])
        return Vector(newCoords, newUp, newDown, N)

def detHoles(o, E, aS, bSq, eta=0.1):
    det = (o + eta*1j + E - aS[-1])
    for i in range(1, len(aS)-1):
        det = o + eta*1j + E - aS[-i-1] - bSq[-i]/det
    return (o + eta*1j + E - aS[0] - bSq[1]/det)

def detParts(o, E, aS, bSq, eta=0.1):
    det = (o + eta*1j - E + aS[-1])
    for i in range(1, len(aS)-1):
        det = o + eta*1j - E + aS[-i-1] - bSq[-i]/det
    return (o + eta*1j - E + aS[0] - bSq[1]/det)

def generateVec0(N, Nu, Nd):
    vec = Vector([0], [0], [0], N)
    listUp, listDown = [], []
    for i in range(2**N):
        if sum(toBinaryList(i, N)) == Nu:
            listUp.append(i)
        if sum(toBinaryList(i, N)) == Nd:
            listDown.append(i)
    for i in range(len(listUp)):
        for j in range(len(listDown)):
            vec += Vector([np.random.rand()], [listUp[i]], [listDown[j]], N)
    return vec
          
#Definición inicial, para N < 2 no tiene sentido.
N = 2
Nu = 1
Nd = 2
tols=[]
tMat = buildMatrix(e0*np.ones(N), t*np.ones(N-1), t*np.ones(N-1))
#tMat[-1][0], tMat[0][-1] = t, t
Ulist = U*np.ones(N)      
#Calculamos el estado fundamental, en este caso directamente con Lanczos
tol = 1e-6
E0 = 0
dimn = 2
dimng = 2
nu = toDecList([1]*Nu)
nd = toDecList([1]*Nd)
Enew = 20**2
vec0 = Vector([1], [nu], [nd], N)
#vec0 = generateVec0(N, Nu, Nd)
t0 = time.time()
while abs(Enew - E0) > tol:
    E0 = Enew
    aS, bSq, vecs = lanczos(vec0, dimn)
    ham = buildMatrix(aS, np.sqrt(bSq)[1::], np.sqrt(bSq)[1::])
    eig = np.linalg.eig(ham)
    Enew = min(eig[0])
    ind = list(eig[0]).index(Enew)
    vec0 = groundState(eig[1][:,ind], vecs)
    tols.append(abs(Enew - E0))
    print(abs(Enew - E0))
tf = time.time()
print(tf - t0)
plt.figure()
plt.plot(np.arange(0, len(tols), 1), tols)
plt.plot(np.arange(0, len(tols), 1), tol*np.ones(len(tols)), "--")
plt.xlabel("Número de pasos")
plt.grid()
plt.ylabel("$|E_{new} - E_{old}|$")
plt.title("U = " + str(U) + ", N = " + str(N))
#plt.savefig("Imágenes/convergenceU8N8.png", dpi=200)

fState = vec0*(vec0.norm()**-0.5)
NP = 1000
G = np.zeros((N, N, NP), dtype="complex")
o = np.linspace(-10+0.5*U, 10+0.5*U, NP)
for i in range(N):
    nP = nParticles(i, fState, "up", N)
    nH = 1 - nP
    initialPart = destructionVec(i, fState, "up", N)*(nP**-0.5)
    #initialHoles = creationVec(i, fState, "up", N)*(nH**-0.5)
    aSp, bSqp = lanczos(initialPart, dimng)[:2:]
    #aSh, bSqh = lanczos(initialHoles, dimng)[:2:]
    G[i,i,:] = nP/detParts(o, E0, aSp, bSqp) #+ nH/detHoles(o, E0, aSh, bSqh)
for i in range(N):
    for j in range(N):
        if j != i:
            nComb = fState.prod(creationVec(i, destructionVec(i, fState, "up", N), "up", N) + creationVec(i, destructionVec(j, fState, "up", N), "up", N) + creationVec(j, destructionVec(i, fState, "up", N), "up", N) + creationVec(j, destructionVec(j, fState, "up", N), "up", N))
            hComb = fState.prod(destructionVec(i, creationVec(i, fState, "up", N), "up", N) + destructionVec(i, creationVec(j, fState, "up", N), "up", N) + destructionVec(j, creationVec(i, fState, "up", N), "up", N) + destructionVec(j, creationVec(j, fState, "up", N), "up", N))
            initState = (destructionVec(i, fState, "up", N) + destructionVec(j, fState, "up", N))*(nComb**-0.5)
            aSp, bSqp = lanczos(initialPart, dimng)[:2:]
            aSh, bSqh = lanczos(initialHoles, dimng)[:2:]
            G[i,j,:] = 0.5*(nP/detParts(o, E0, aSp, bSqp) + nH/detHoles(o, E0, aSh, bSqh) - G[i,i,:] - G[j,j,:])

def trapecio(x, y):
    dx = x[1] - x[0]
    sum = 0
    for i in range(1, len(y)):
        sum += 0.5*dx*(y[i-1] + y[i])
    return sum

eTB = np.linalg.eig(tMat)[0]
dos = -np.imag(np.trace(G, axis1=0, axis2=1))/np.pi
re = np.real(np.trace(G, axis1=0, axis2=1))
print(trapecio(o, dos))
print(Enew)

plt.figure()
plt.plot(o-0.5*U*np.ones(NP), dos)
plt.grid()
plt.xlabel("$\\left(\\omega-\\frac{U}{2}\\right)(t)$")
plt.ylabel("$-\\frac{1}{\\pi}Im\\left(G(\\omega)\\right)$")
plt.ylim(-0.1, max(dos)+0.1)
for e in eTB:
    plt.vlines(e, -0.1, max(dos)+0.1, ls="--")
#plt.savefig("Imágenes/lanczosHubbard2NegU8Imag.png", dpi=200)
plt.figure()
plt.plot(o-0.5*U*np.ones(NP), re)
plt.grid()
plt.xlabel("$\\left(\\omega-\\frac{U}{2}\\right)(t)$")
plt.ylabel("$Re\\left(G(\\omega)\\right)$")
#plt.savefig("Imágenes/lanczosHubbard6U10Real.png", dpi=200)

a = 1
k = np.linspace(-np.pi/a, np.pi/a, NP)

plt.figure()
plt.plot(k, e0 - 2*t*np.cos(k*a))
plt.grid()
plt.xlabel("k")
plt.ylabel("E")
plt.title("Banda en un modelo tight-binding simple")
#plt.savefig("Imágenes/BandTB.png", dpi=200)

E = np.linspace(-2, 2, NP)

plt.figure()
plt.plot(E, (1/(np.pi*a))*(1/np.sqrt(1-(E/2)**2)))
plt.grid()
plt.xlabel("E(t)")
plt.ylabel("g(E)")
plt.title("Densidad de estados en un modelo tight-binding")
#plt.savefig("Imágenes/DosTB.png", dpi=200)

Ns = 100
Nss = 5
t = np.linspace(0.01, 0.1, Nss)
Us = np.linspace(-3, 3, Ns)
probs = []
for k in range(Nss):
    tMat = buildMatrix(e0*np.ones(N), t[k]*np.ones(N-1), t[k]*np.ones(N-1))
    tMat[-1][0], tMat[0][-1] = t[k], t[k]
    ps = np.zeros(Ns)
    for i in range(Ns):
        print(i, k)
        Enew = -1
        Ulist = Us[i]*np.ones(N)
        vec0 = Vector([1], [nu], [nd], N)
        while abs(Enew - E0) > tol:
            E0 = Enew
            aS, bSq, vecs = lanczos(vec0, dimn)
            ham = buildMatrix(aS, np.sqrt(bSq)[1::], np.sqrt(bSq)[1::])
            eig = np.linalg.eig(ham)
            Enew = min(eig[0])
            ind = list(eig[0]).index(Enew)
            vec0 = groundState(eig[1][:,ind], vecs)
        fState = vec0*(vec0.norm()**-0.5)
        for j in range(N):
            ps[i] += nnParticles(j, j, "up", "down", fState, N)
    probs.append(ps)

t[2] = round(t[2], 3)
plt.figure()
plt.xlabel("$\\frac{U}{t}$")
plt.ylabel("$P = \\sum_i\\langle\\psi_0|c_{i\\downarrow}^{\\dagger}c_{i\\uparrow}^{\\dagger}c_{i\\uparrow}c_{i\\downarrow}|\\psi_0\\rangle$")
for i in range(len(t)):
    plt.plot(Us, probs[i], label="t = " + str(t[i]))
plt.grid()
plt.legend()
plt.savefig("Imágenes/ProbabilityPairsAll4.png", dpi=200)

plt.show()