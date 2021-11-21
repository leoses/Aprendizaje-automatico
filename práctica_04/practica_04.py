import numpy as np
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from displayData import displayData
from checkNNGradients import checkNNGradients


def func_sigmoid(X):
    return (1 / (1 + np.exp(-X)))
  
def calcula_propagacion(X, theta1, theta2):
    m = np.shape(X)[0]
    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, theta1.T)
    A2 = np.hstack([np.ones([m, 1]), func_sigmoid(Z2)])
    Z3 = np.dot(A2, theta2.T)
    A3 = func_sigmoid(Z3)
    return A1, A2, A3

def coste(X, y, t1, t2):
    m = np.shape(X)[0]
    a1, a2, h = calcula_propagacion(X, t1, t2)
    return (1/m)*np.sum(-y*np.log(h) - (1-y)*np.log(1-h))

def coste_reg(X, y, t1, t2, reg):
    m = np.shape(X)[0]
    #En el coste reg no hay que incluir la primera columna de t1 y t2
    otra = (reg/(2*m)) * (np.sum(np.power(t1[1:], 2)) + np.sum(np.power(t2[1:], 2)))
    return coste(X, y, t1, t2) + otra


def gradiente_regularizacion(gradiente, m, reg, theta):
    columnaGuardada = gradiente[0]
    gradiente = gradiente + (reg/m)*theta
    gradiente[0] = columnaGuardada
    return gradiente

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):

    m = np.shape(X)[0]

    Theta1 = np.reshape(
        params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(
        params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas+1)))

    # Calculamos propagacion
    A1, A2, H = calcula_propagacion(X, Theta1, Theta2)

    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    for t in range(m):
        a1t = A1[t, :] # (401,)
        a2t = A2[t, :] # (26,)
        ht = H[t, :] # (26,)
        yt = Y[t] # (10,)

        d3t = ht - yt # (10,)
        d2t = np.dot(Theta2.T, d3t)*(a2t*(1-a2t)) # (10,)

        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    gradiente1 = Delta1/m
    gradiente2 = Delta2/m

    gradiente1 = gradiente_regularizacion(gradiente1, m, reg, Theta1)
    gradiente2 = gradiente_regularizacion(gradiente2, m, reg, Theta2)

    return coste_reg(X, Y, Theta1, Theta2, reg), np.concatenate([np.ravel(gradiente1), np.ravel(gradiente2)])


def y_oneHot(y, numLabels):
    m = len(y)

    y = (y-1)
    y_onehot = np.zeros((m, numLabels))

    for i in range(m):
        y_onehot[i][y[i]] = 1

    return y_onehot


def optimize_backprop_and_check (t1, t2,num_entradas, num_ocultas, num_etiquetas, reg, X, y_onehot, laps, Y_original):

    eini = 0.12
    params = np.concatenate([np.ravel(t1),np.ravel(t2)])
    pesos = np.random.uniform(-eini,eini, params.shape[0])

    out = minimize(fun = backprop, x0 = pesos, args = (num_entradas, num_ocultas, num_etiquetas, X, y_onehot, reg), method='TNC', jac = True, options = {'maxiter': laps})

    Thetas1 = out.x[:(num_ocultas*(num_entradas+1))].reshape(num_ocultas,(num_entradas+1))
    Thetas2 = out.x[(num_ocultas*(num_entradas+1)):].reshape(num_etiquetas,(num_ocultas+1))
    
    A1,A2,H = calcula_propagacion(X,Thetas1, Thetas2)
    
    maxIndices = np.argmax(H,axis=1) + 1
   
    acertados = np.sum(maxIndices==Y_original.ravel())
    print("Porcentaje acertados: " + str(acertados*100/np.shape(H)[0]) + "%")


data = loadmat("ex4data1.mat")

X = data['X']
y_original = data['y']  # (5000,1)

#Grafica del apartado 1
#amount = 100
#sample = np.random.choice(X.shape[0],amount)
#res = displayData(X[sample])
#plt.savefig("ImagenNum.png")
#plt.show()

num_etiquetas = 10
num_ocultas = 25
num_entradas = np.shape(X)[1]

y = y_oneHot(y_original.ravel(), num_etiquetas)

weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

params_rn = np.concatenate([np.ravel(theta1), np.ravel(theta2)])

cost, gradiente = backprop(
    params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, 1)
    
print(np.sum(checkNNGradients(backprop, 1)))
print("Coste con regularizacion: " + str(cost))

optimize_backprop_and_check(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, 1, X, y, 70, y_original)