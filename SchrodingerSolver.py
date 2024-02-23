import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

meshno = 1000
domain = [-10, 10]
#dx = (domain[1]-domain[0])/meshno

xvals = np.linspace(domain[0], domain[1], meshno)
dx = xvals[1]-xvals[0]
m = 1
a = 2

#create the differentiation matrix to a factor of -1
diffmat = np.ones(meshno)
diffmat = 2*np.diag(diffmat)
diffmat += -1*np.diag(np.ones(meshno-1), -1)
diffmat += -1*np.diag(np.ones(meshno-1), 1)

#defines the potential
def potFun(x):
    v0 = a/2*x**2
    #if x < 0:
    #    return 1e12
    #else:
    #    return a*x
    return v0

#analytical result for the harmonic oscillator
def trueState(x, n):
    B = (m*a)**(1/4)
    coeff = np.sqrt(B/(2**n * sp.special.factorial(n) *np.sqrt(np.pi)))
    functional = sp.special.hermite(n)
    return coeff * functional(x) * np.exp(- (B*x)**2/2)


#create the potential array
potvals = np.array([potFun(v) for v in xvals])
potmat = np.diag(potvals)



#create the Hamiltonian operator
Hmat = 1/(2*m*dx**2) * diffmat + potmat
print(Hmat)

#solve for eigenvalues and eigenvectors
evals, evecs = np.linalg.eig(Hmat)
#sort by eigenvalues
sortnos = np.argsort(evals)
evals = evals[sortnos]
evecs = evecs[:, sortnos]

#plot the first 4
fig, ax = plt.subplots(4, 1)
for i in range(4):
    print(evals[i])
    ax[i].plot(xvals, evecs[:,i])
    #ax[i].plot(xvals, trueState(xvals, i)/max(trueState(xvals, i))*max(abs(evecs[:,i])))
    ax[i].plot(xvals, potvals/max(potvals) *max(abs(evecs[:,i])))
plt.show()