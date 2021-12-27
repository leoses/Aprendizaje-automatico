import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def gradiente(Theta,X,Y):
	m = np.shape(X)[0]
	H = np.dot(X,Theta)
	return np.dot(H-Y.T,X)/m

def gradiente_reg_1(theta, X, Y, landa):
    theta = theta.reshape((theta.shape[0], 1))
    m = X.shape[0]
    H = np.matmul(X, theta)
    grad = (X.T.dot(H-Y)/(m)) + ((landa)/m)*theta
    return grad

def gradiente_reg_2(theta, X, Y, landa):
    m = np.shape(X)[0]
    grad = gradiente(theta,X,Y)
    grad_0 = grad[0]
    j = grad + (landa/(m))*theta
    j[0] = grad_0
    return j


def coste_reg(theta, X, Y, landa):
    m = np.shape(X)[0]
    H = np.dot(X,theta)
    return (np.sum((H-Y.T)**2))/(2*m) + (landa/(2*m))*np.sum(theta[1:] **2)


def calcula_coste_gradiente_reg_1(theta, X, Y, landa):
    return coste_reg(theta, X, Y, landa), gradiente_reg_1(theta, X, Y, landa)


def calcula_coste_gradiente_reg_2(theta, X, Y, landa):
    return coste_reg(theta, X, Y, landa), gradiente_reg_2(theta, X, Y, landa)


def normaliza(matriz):
    #xi = (xi - ui) / si
    #ui = media ; si = desviación típica
    matriz_normal = np.empty_like(matriz)

    u = np.mean(matriz, axis=0)
    s = np.std(matriz, axis=0)
    # print('Media: ' + str(u))
    # print('Desv: ' + str(s))

    matriz_normal = (matriz - u) / s

    return [matriz_normal, u, s]

def nuevos_datos(X, p) :
    mat = np.zeros([np.shape(X)[0], p])

    for i in range(1, p+1) :
        mat[:, i-1] = (X**i).ravel()
    return mat


def apartado1(X, y):
    #m = np.shape(X)[0]

    auxX = np.insert(X, 0, 1, axis=1)

    theta = np.array([[1], [1]])
    landa = 0

    #coste = coste_reg(theta, auxX, y, landa)
    #g = gradiente_reg_1(theta, auxX, y, landa)

    result = minimize(calcula_coste_gradiente_reg_1, theta,
                      args=(auxX, y, landa), jac=True, method='TNC')

    # Pintamos grafica
    plt.figure()
    # Pintamos x
    plt.plot(X, y, "x", color='red')

    min_x = min(X)
    max_x = max(X)
    min_y = result.x[0] + result.x[1]*min_x
    max_y = result.x[0] + result.x[1]*max_x

    # Pintamos recta
    plt.plot([min_x, max_x], [min_y, max_y], color='blue', linewidth=2)
    # Guardamos imagen
    plt.savefig("regresionLineal.png")
    plt.show()
    plt.close()


def apartado2(X, y, Xval, yval, landa):
    m = np.shape(X)[0]
    mValidacion = np.shape(Xval)[0]
    Xval = np.hstack([np.ones([mValidacion,1]),Xval])
    
    eValidation = np.zeros([m])
    eTraning = np.zeros([m])
    
    for i in range(1,m+1):
        theta = np.zeros(np.shape(X)[1])
        result = minimize(calcula_coste_gradiente_reg_2, theta, args = (X[0:i], y[0:i,0], landa), jac = True, method = 'TNC')
        eTraning[i-1] = coste_reg(result.x, X[0:i], y[0:i], landa)
        eValidation[i-1] = coste_reg(result.x, Xval, yval, landa)
    
    plt.figure()
    plt.plot(np.linspace(0,11,12,dtype=int), eTraning, label='Train')
    plt.plot(np.linspace(0,11,12,dtype=int), eValidation, label='Cross Validation')
    plt.legend()
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')    
    plt.savefig("curvasAprendizaje.png")
    plt.show()
    # Guardamos imagen
    plt.close()


def apartado3(X, p, landa) :
    mat = nuevos_datos(X, p)
    u = np.zeros([p])
    s = np.zeros([p])
    mat, u, s = normaliza(mat)

    mat = np.hstack([np.ones([np.shape(mat)[0],1]),mat])

    theta = np.zeros(np.shape(mat[1]))

    result = minimize(calcula_coste_gradiente_reg_2, theta,
                      args=(mat, y, landa), jac=True, method='TNC')

    # Pintamos grafica
    plt.figure()
    # Pintamos x
    plt.plot(X, y, "x", color='red')
    # Pintamos la linea
    lineX = np.arange(np.min(X),np.max(X),0.05)
    aux_x = (nuevos_datos(lineX, p)-u) / s
    lineY = np.hstack([np.ones([len(aux_x),1]),aux_x]).dot(result.x)
    plt.plot(lineX, lineY, '-', c = 'blue')
    plt.savefig("regPolinomial.png")
    plt.show()
    # Guardamos imagen
    plt.close()


data = loadmat('ex5data1.mat')

y = data['y']
X = data['X']
#auxX = np.insert(X, 0,1, axis=1)
m = np.shape(X)[0]
auxX = np.hstack([np.ones([m,1]),X])

Xval = data['Xval']
yval = data['yval']
Xval = data["Xval"]
#Xval = np.insert(Xval,0,1,axis=1)

apartado1(X,y)
#apartado2(auxX, y, Xval, yval, 0)
#apartado3(X, 8, 0)