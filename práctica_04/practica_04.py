import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from displayData import displayData;

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
def coste_sin_reg(X, y, t1, t2):
    m = np.shape(X)[0]
    a1,a2,h = calcula_propagacion(X,t1,t2)
    part1 = y*np.log(h)
    part2 = (1-y)*np.log(1-h+1e-9)
    return (-1/m)*np.sum(part1+part2)

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
def func_coste_reg(X,y, t1,t2, reg):
   m = np.shape(X)[0]
   aux = coste_sin_reg(X, y, t1, t2)
   otra = (reg/(2*m)) * (np.sum(np.power(t1[1:],2)) +np.sum(np.power(t2[1:], 2)))
   return aux + otra

def gradiente_reg(theta, X,Y, lamb):
    m = len(X)
    return gradiente(theta, X, Y) + lamb/m * theta

def getEtiqueta(Y, etiqueta):
    #aplanamos matriz y, y la comparamos elemento a elemento con la etiqueta
    etiquetas = (np.ravel(Y) == etiqueta) 
    # Devolvemos vector con elems pertenecientes a la clase (si ==  0no pertenece, si == 1 si)
    return etiquetas * 1

def calcula_propagacion(X, theta1, theta2):
    m = np.shape(X)[0]
    A1 = np.hstack([np.ones([m,1]),X])
    Z2 = np.dot(A1, theta1.T)
    A2 = np.hstack([np.ones([m,1]),func_sigmoid(Z2)])
    Z3 = np.dot(A2, theta2.T)
    A3 = func_sigmoid(Z3)
    return A1, A2, A3

def gradiente_regularizacion(gradiente, m, reg, theta):
	columnaGuardada = gradiente[0]
	gradiente = gradiente + (reg/m)*theta
	gradiente[0] = columnaGuardada
	return gradiente

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	m = np.shape(X)[0]

	Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)] , (num_ocultas, (num_entradas+1)))
	Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):] , (num_etiquetas, (num_ocultas+1)))

	Delta1 = np.zeros_like(Theta1)
	Delta2 = np.zeros_like(Theta2)

	A1, A2, H = calcula_propagacion(X,Theta1, Theta2)

	for t in range(m):
		a1t = A1[t,:]
		a2t = A2[t,:]
		ht = H[t,:]
		yt = y[t]

		d3t = ht -yt
		d2t = np.dot(Theta2.T, d3t)*(a2t*(1-a2t))

		Delta1 = Delta1 + np.dot(d2t[1:,np.newaxis], a1t[np.newaxis,:])
		Delta2 = Delta2 + np.dot(d3t[:,np.newaxis],a2t[np.newaxis,:])

	gradiente1 = Delta1/m
	gradiente2 = Delta2/m

	gradiente1 = gradiente_regularizacion(gradiente1, m, reg,Theta1)
	gradiente2 = gradiente_regularizacion(gradiente2, m, reg,Theta2)

	return func_coste_reg(X,y,Theta1,Theta2,reg), np.concatenate([np.ravel(gradiente1),np.ravel(gradiente2)])

# PRACTICA 4 - Apartado 1
#def randomTheta():


data = loadmat("ex4data1.mat")
# se pueden consultar las claves con data.keys()
y = data['y'].ravel() # (5000,1) -> (5000,)
X = data['X']
num_etiquetas = 10
num_ocultas = 25

m = len(y)
input_size = np.shape(X)[1]

y = (y-1)
y_onehot = np.zeros((m,num_etiquetas)) # (5000,10)

for i in range(m):
    y_onehot[i][y[i]] =  1

theta = np.zeros((num_etiquetas,X.shape[1]))

#Esto era la grafica creo :D
#coste_por_ej = coste(theta,X,y_onehot) 
#print("Coste porfa plis funca: " + str(coste_por_ej))

#sample = np.random.choice(X.shape[0] , num_etiquetas)
#plt.imshow(X[sample,:].reshape(-1,20).T)
#fig , axes = displayData(X[sample,:])

#plt.show()

weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
	
num_entradas = np.shape(X)[1]

params_rn = np.concatenate([np.ravel(theta1),np.ravel(theta2)])

costeee, coso = backprop(params_rn,num_entradas,num_ocultas,num_etiquetas,X,y,1)

print(costeee)
## Theta1 es de dimensión 25 x 401
## Theta2 es de dimensión 10 x 26
#
#esperable = np.zeros(num_etiquetas)
#for r in range(num_etiquetas):
#    esperable[r] = np.argmax(calculaPropagacion(X[sample[r],:],theta1,theta2))
#
#numAciertos = 0
#for r in range(X.shape[0]):
#    resultado = np.argmax(calculaPropagacion(X[r,:],theta1,theta2))
#    if(resultado+1) == y[r]:
#        numAciertos +=1
#print("Porcentaje de aciertos: " + str(numAciertos / X.shape[0]*100) + "%")