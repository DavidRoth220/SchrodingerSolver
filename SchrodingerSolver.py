import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

meshno = 1000
domain = [-10, 10]
dx = (domain[1]-domain[0])/meshno

xvals = np.linspace(domain[0], domain[1], meshno)

m = 1
a = 1


diffmat = np.ones(meshno)
diffmat = 2*np.diag(diffmat)
diffmat += -1*np.diag(np.ones(meshno-1), -1)
diffmat += -1*np.diag(np.ones(meshno-1), 1)
print(diffmat)

def potFun(x):
    #v0 = a/2*x**2
    if x < 0:
        return 1e9
    else:
        return a*x
    return v0

def trueState(x, n):
    B = (m*a)**(1/4)
    coeff = np.sqrt(B/(2**n * sp.special.factorial(n) *np.sqrt(np.pi)))
    functional = sp.special.hermite(n)
    return coeff * functional(x) * np.exp(- (B*x)**2/2)

potvals = np.array([potFun(v) for v in xvals])
potmat = np.diag(potvals)

print(potmat)

Hmat = 1/(2*m*dx**2) * diffmat + potmat

evals, evecs = np.linalg.eig(Hmat)
sortnos = np.argsort(evals)
evals = evals[sortnos]
evecs = evecs[:, sortnos]


fig, ax = plt.subplots(5, 1)
for i in range(5):
    print(evals[i])
    ax[i].plot(xvals, evecs[:,i])
    #ax[i].plot(xvals, trueState(xvals, i)/max(trueState(xvals, i))*max(abs(evecs[:,i])))
    ax[i].plot(xvals, potvals/max(potvals) *max(abs(evecs[:,i])))
plt.show()