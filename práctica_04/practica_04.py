import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from displayData import displayData;

# Aplicamos formula de la funcion sigmoide tal cual
# np.exp hace e^X siendo x lo que le pases como parametro
def func_sigmoid(X):
    return (1 / (1 + np.exp(-X)))

def calcula_propagacion(X, theta1, theta2):
    m = np.shape(X)[0]
    A1 = np.hstack([np.ones([m,1]),X])
    Z2 = np.dot(A1, theta1.T)
    A2 = np.hstack([np.ones([m,1]),func_sigmoid(Z2)])
    Z3 = np.dot(A2, theta2.T)
    A3 = func_sigmoid(Z3)
    return A1, A2, A3    

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


def getEtiqueta(Y, etiqueta):
    #aplanamos matriz y, y la comparamos elemento a elemento con la etiqueta
    etiquetas = (np.ravel(Y) == etiqueta) 
    # Devolvemos vector con elems pertenecientes a la clase (si ==  0no pertenece, si == 1 si)
    return etiquetas * 1

def coste(X,y, t1,t2):
	m = np.shape(X)[0]
	a1,a2,h = calcula_propagacion(X,t1,t2)
	return (1/m)*np.sum(-y*np.log(h) - (1-y)*np.log(1-h))


def costeSinRegularizar(X,y, t1,t2):
    m = np.shape(X)[0]
    a1,a2,h = calcula_propagacion(X,t1,t2)
    part1 = y*np.log(h)
    part2 = (1-y)*np.log(1-h+1e-9)
    return (-1/m)*np.sum(part1+part2)

def coste_reg(X,y, t1,t2, reg):
	m = np.shape(X)[0]	
	aux = costeSinRegularizar(X, y, t1, t2)
	otra = (reg/(2*m)) * (np.sum(np.power(t1[1:],2)) +np.sum(np.power(t2[1:], 2)))
	
	return aux + otra

def gradiente_regularizacion(gradiente, m, reg, theta):
	columnaGuardada = gradiente[0]
	gradiente = gradiente + (reg/m)*theta
	gradiente[0] = columnaGuardada
	return gradiente

def matrixY(Y, nEtiquetas):
    nY = np.zeros((len(Y), nEtiquetas))
    yaux = np.array(Y) -1
    
    for i in range(len(nY)):
        z = yaux[i]
        if(isinstance(z, np.uint8)):
            if(z == 10): z = 0
            nY[i][z] = 1
        else:
            z = yaux[i].all()
            if(z == 10): z = 0
            nY[i][z] = 1
            
    return nY

def backprop(params_rn, num_entradas,num_ocultas, num_etiquetas, X, Y, reg):
    
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)] , (num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):] , (num_etiquetas, (num_ocultas+1)))

    #print(Theta1.shape)
    #print(Theta2.shape)
    nMuestras = len(X)
    
    #Calculamos propagacion
    A1, A2, H = calcula_propagacion(X, Theta1, Theta2)
    
    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)
    
    for t in range(nMuestras):
        a1t = A1[t,:]
        a2t = A2[t,:]
        ht = H[t,:]
        yt = y[t]
        
        d3t = ht -yt
        d2t = np.dot(Theta2.T, d3t)*(a2t*(1-a2t))
        Delta1 = Delta1 + np.dot(d2t[1:,np.newaxis], a1t[np.newaxis,:])
        Delta2 = Delta2 + np.dot(d3t[:,np.newaxis],a2t[np.newaxis,:])
    
    gradiente1 = Delta1/nMuestras
    gradiente2 = Delta2/nMuestras
    
    G1 = gradiente_regularizacion(gradiente1, nMuestras, reg,Theta1)
    G2 = gradiente_regularizacion(gradiente2, nMuestras, reg,Theta2)
    
    return coste_reg(X,Y,Theta1,Theta2, reg), np.concatenate((G1, G2), axis = None)

def y_oneHot(y,numLabels):
	m = len(y)
	
	y = (y-1)
	y_onehot = np.zeros((m,numLabels))

	for i in range(m):
		y_onehot[i][y[i]] = 1

	return y_onehot

data = loadmat("ex4data1.mat")
# se pueden consultar las claves con data.keys()
X = data['X']
y = data['y'].ravel() # (5000,1) -> (5000,)

num_etiquetas = 10
num_ocultas = 25

m = len(y)
input_size = np.shape(X)[1]

y_onehot = y_oneHot(y, num_etiquetas)
#print(y_onehot.shape)

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

params_rn = np.hstack([np.ravel(theta1),np.ravel(theta2)])

costeee, coso = backprop(params_rn,num_entradas,num_ocultas,num_etiquetas,X,y_onehot,1)

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