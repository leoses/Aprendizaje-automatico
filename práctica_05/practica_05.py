import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def gradiente_reg(theta, X, Y, landa):
    theta = theta.reshape((theta.shape[0], 1))
    m = X.shape[0]
    H = np.matmul(X, theta)
    grad = (X.T.dot(H-Y)/(m)) + ((landa)/m)*theta
    return grad


def coste_reg(theta, X, Y, landa):
    m = X.shape[0]
    H = np.matmul(X, theta).reshape((m, 1))
    return float(np.dot((H-Y).T, (H-Y)/(2*m))) + (landa/(2*m)) * float(theta[1:].T.dot(theta[1:]))


def calcula_coste_gradiente_reg(theta, X, Y, landa):
    return coste_reg(theta, X, Y, landa), gradiente_reg(theta, X, Y, landa)


def apartado1(X, y):
    m = np.shape(X)[0]

    auxX = np.insert(X, 0, 1, axis=1)

    theta = np.array([[1], [1]])
    landa = 0

    coste = coste_reg(theta, auxX, y, landa)
    g = gradiente_reg(theta, auxX, y, landa)

    result = minimize(calcula_coste_gradiente_reg, theta,
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
    theta = np.ones((2, 1))
    muestras, trainVector, valVector = [], [], []
    for x in range(1, 13, 1):
        # vamos seleccionando primer conjunto entrenamiento, luego segundo...
        train_aux = X[:x, :]
        y_aux = y[:x]
        fit_theta = minimize(calcula_coste_gradiente_reg, theta, args=(auxX, y, landa), jac=True, method='TNC').x
        trainVector.append(coste_reg(fit_theta, train_aux, y_aux,landa))
        valVector.append(coste_reg(fit_theta, Xval, yval, landa))
        muestras.append(y_aux.shape[0])
    
    plt.figure()
    plt.plot(muestras, trainVector, label='Train')
    plt.plot(muestras, valVector, label='Cross Validation')
    plt.legend()
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')    
    plt.show()


data = loadmat('ex5data1.mat')

y = data['y']
X = data['X']
auxX = np.insert(X, 0,1, axis=1)
Xval = data['Xval']
yval = data['yval']
Xval = np.insert(Xval,0,1,axis=1)

# apartado1(X,y)
apartado2(auxX,y,Xval, yval,0)
