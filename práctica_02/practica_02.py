import numpy as np
from numpy.core.fromnumeric import transpose
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import optimize
from scipy.optimize import fmin_tnc

#metodo de lectura de ficheros csv
def carga_csv(s):
    valores = read_csv (s , header=None).to_numpy()
    return valores.astype(float)

# Apartado 1.1, pintado de la grafica con los datos recibidos
def grafica_v1(X, Y):
    # creamos figura
    fig = plt.figure()
    # Adding axes on the figure
    ax = fig.add_subplot(111)
    
    # Obtiene un vector con los índices de los ejemplos positivos (1 en reg logistica)
    pos = np.where (Y == 1 )
    # Obtener vector con los indices de los ejemplos negativos (0 en reg logistica)
    neg = np.where(Y == 0)
    
    # Dibuja los ejemplos positivos
    plt.scatter(X[ pos , 0 ] , X[ pos , 1 ] , marker='+' , c='k', label = "Admited")
    # Dibuja los ejemplos negativos
    plt.scatter(X[ neg, 0 ] , X[ neg , 1 ] , marker='.' , c='orange', label = "Not Admitted")

    # Anadimos leyenda a la grafica y la posicionamos bien
    plt.legend(loc = 'upper right')

    # Anadimos el texto a los ejes (como en la grafica que aparece de ejemplo)
    ax.set_xlabel('Exam1 Score', fontsize=10)
    ax.set_ylabel('Exam2 Score', fontsize=10)

    # Guardamos en formato png
    plt.savefig('regresionLogistica_v1.png')
    # Mostramos por pantalla
    plt.show()
    # Cerramos
    plt.close()

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
def gradiente(theta,X,Y,):
    H = func_sigmoid(np.matmul(X,theta))
    return 1/len(X) * (np.matmul(np.transpose(X), H-Y))

# Apartado 1.4, calculo del valor optimo de parametros
def optimizacion(X,Y):
    n=np.shape(X)[1]
    # Inicializamos todos los valores de theta a 0  
    theta=np.zeros(n) 
    result = fmin_tnc(coste_vec,theta , gradiente , args =(X, Y))
    return result[0]

# Metodo para pintar la frontera
def pinta_frontera(X,Y,theta):

    # Creamos grafica
    fig = plt.figure()
    
    # Adding axes on the figure
    ax = fig.add_subplot(111)

    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = func_sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta_opt))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    # Obtiene un vector con los índices de los ejemplos positivos (1 en reg logistica)
    pos=np.where(Y == 1)
    # Obtiene un vector con los índices de los ejemplos negativos (0 en reg logistica)
    neg=np.where(Y == 0)

    # Dibuja los ejemplos positivos
    plt.scatter(X[ pos , 1 ] , X[ pos , 2 ] , marker='+' , c='k', label = "Admited")
    # Dibuja los ejemplos negativos
    plt.scatter(X[ neg, 1 ] , X[ neg , 2 ] , marker='.' , c='orange', label = "Not Admitted")

    # Anadimos leyenda a la grafica y la posicionamos bien
    plt.legend(loc = 'upper right')

    # Anadimos el texto a los ejes (como en la grafica que aparece de ejemplo)
    ax.set_xlabel('Exam1 Score', fontsize=10)
    ax.set_ylabel('Exam2 Score', fontsize=10)

    # Ajustamos los rangos en x y en y de los datos para que la grafica sea como la de ejemplo
    plt.xlim([28, 102])
    plt.ylim([28, 102])

    # guardamos grafica
    plt.savefig("frontera.png")
    # mostramos grafica
    plt.show()
    # Cerramos
    plt.close()

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


datos = carga_csv('ex2data1.csv')
X = datos[:, :-1]
Y = datos[:, -1]

# Apartado 1.1
# ----------------------------------------------------------
# grafica_v1(X,Y)

# Anadimos columna de 1s
X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
n=np.shape(X)[1]

# Apartado 1.2 & 1.3
# ----------------------------------------------------------
# Inicializamos todos los valores de theta a 0
#theta=np.zeros(n) 

# Calculamos coste
#cost = coste_vec(theta,X,Y)
#cost_2 = coste(theta,X,Y)

#print("Coste estimado vectorizado: " + str(cost))
#print("Coste estimado sin vectorizar: " + str(cost_2))
#print(gradiente(theta,X,Y))

# Apartado 1.4, calculo de theta optima para minimizar funcion de coste
# ----------------------------------------------------------
theta_opt = optimizacion(X,Y)

# Calculamos coste
cost = coste_vec(theta_opt,X,Y)
# Mostramos por pantalla para confirmar que el valor es el esperado
print("Coste minimo con theta optimizada: " + str(cost))

# Pintamos la frontera
pinta_frontera(X,Y,theta_opt)

# Apartado 1.5, calculo del porcentaje de ejemplos clasificados correctamente
# ----------------------------------------------------------
print("Porcentaje de ejemplos calculados correctamente: " + str(calcula_porcentaje(X,Y,theta_opt)) + "%")