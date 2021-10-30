import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc


# Apartado 1.2, funcion sigmoide
# Aplicamos formula de la funcion sigmoide tal cual
# np.exp hace e^X siendo x lo que le pases como parametro
def func_sigmoid(X):
    return (1 / (1 + np.exp(-X)))

# Apartado 1.3, funcion de coste
# Version sin vectorizar aplicando formula
def coste(theta,X,Y):
    m = len(X)
    H = func_sigmoid(np.matmul(X,transpose(theta)))
    return 1/m * np.sum(-(Y * np.log(H)) - (1 - Y) * np.log(1 - H))


# Version vectorizada del metodo de coste
# Aplicamos formula
def coste_vec(theta,X,Y):
    m = len(X)
    H = func_sigmoid(np.matmul(X, theta)) # h = X * theta
    transpose_log_h = np.transpose(np.log(H))
    transpose_log_one_minus_h = np.transpose(np.log(1 - H))
    return -(1/m) * (np.matmul(transpose_log_h , Y) + np.matmul(transpose_log_one_minus_h , (1-Y)))


# Version vectorizada del calculo de gradiente
def gradiente(theta,X,Y):
    H = func_sigmoid(np.matmul(X,theta))
    return 1/len(X) * (np.matmul(np.transpose(X), H-Y))

# Apartado 1.5, calculo del porcentaje de ejemplos clasificados correctamente
def calcula_porcentaje(X,Y,theta):
    # Calculamos los valores estimados segun la theta que hemos obtenido
    sig = func_sigmoid(np.matmul(X,theta))

    # Comparamos si nuestra estimación es correcta en comparacion con las Y
    # y devolvemos el numero de ejemplos estimados correctamente
    # if(sig >= 0.5) -> devuelve 1 y si no devuelve 0
    # lo comparamos con la y de los casos de entrenamiento y si corresponde
    # lo hemos calculado bien
    ev_correct = np.sum((sig >= 0.5) == Y)
    
    # Devolvemos el porcentaje
    return ev_correct/len(sig) * 100



# Apartado 2.2
def func_coste_reg(theta,X,Y, lamb):
    m = len(X)
    return coste_vec(theta, X, Y) + lamb/2*m * np.sum(theta*theta)

def gradiente_reg(theta, X,Y, lamb):
    m = len(X)
    return gradiente(theta, X, Y) + lamb/m * theta

def getEtiqueta(Y, etiqueta):
    #aplanamos matriz y, y la comparamos elemento a elemento con la etiqueta
    etiquetas = (np.ravel(Y) == etiqueta) 
    # Devolvemos vector con elems pertenecientes a la clase (si ==  0no pertenece, si == 1 si)
    return etiquetas * 1

# APARTADO 1.2 -> resultado 94.13%
def oneVsAll(X, Y, num_etiquetas, reg):
    """
    oneVsAll entrena varios clasificadores por regresión logística con término
    de regularización 'reg' y devuelve el resultado en una matriz, donde
    la fila i-ésima corresponde al clasificador de la etiqueta i-ésima
    """
    m = X.shape[1]
    theta = np.zeros((num_etiquetas, m))
    y_etiquetas = np.zeros((y.shape[0], num_etiquetas))

    for i in range(num_etiquetas):
        y_etiquetas[:,i] = getEtiqueta(y, i)
    #Cambiamos el 0 por 10 porque si no, todos los valores nos darian 0
    y_etiquetas[:,0] = getEtiqueta(y, 10)

    for i in range(num_etiquetas):
        theta[i, :] = fmin_tnc(func=func_coste_reg, x0=theta[i,:], fprime=gradiente_reg, args=(X, y_etiquetas[:,i], reg))[0]

    resultados = np.zeros(num_etiquetas)
    for i in range(num_etiquetas):
        resultados[i] = calcula_porcentaje( X, y_etiquetas[:,i],theta[i,:])
    print("Evaluacion: ", resultados)
    print("Evaluacion media: ", resultados.mean())

def calculaPropagacion(x, theta1, theta2):
    auxX = x
    s1 = func_sigmoid(np.matmul(theta1, np.insert(auxX, 0, 1)))
    return func_sigmoid(np.matmul(theta2, np.insert(s1, 0, 1)))


data = loadmat("ex3data1.mat")
# se pueden consultar las claves con data.keys()
y = data['y']
X = data['X']
num_etiquetas = 10

oneVsAll(X,y, num_etiquetas, 0.1)

# Selecciona aleatoriamente 10 ejemplos y los pinta
sample = np.random.choice(X.shape[0] , num_etiquetas)
plt.imshow(X[sample,:].reshape(-1,20).T)
plt.axis('off')
plt.show()

weights = loadmat ( 'ex3weights.mat')
theta1 , theta2 = weights['Theta1'] , weights ['Theta2']
# Theta1 es de dimensión 25 x 401
# Theta2 es de dimensión 10 x 26

esperable = np.zeros(num_etiquetas)
for r in range(num_etiquetas):
    esperable[r] = np.argmax(calculaPropagacion(X[sample[r],:],theta1,theta2))

numAciertos = 0
for r in range(X.shape[0]):
    resultado = np.argmax(calculaPropagacion(X[r,:],theta1,theta2))
    if(resultado+1) == y[r]:
        numAciertos +=1
print("Porcentaje de aciertos: " + str(numAciertos / X.shape[0]*100) + "%")



