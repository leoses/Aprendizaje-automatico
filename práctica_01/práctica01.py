import numpy as np
from numpy.core.numeric import empty_like
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(s):
    valores = read_csv (s , header=None).to_numpy()
    return valores.astype(float)

def regresion_lineal_1():
    datos = carga_csv('ex1data1.csv')

    ##Primera columna
    X = datos[:,0]
    ##Segunda columna
    Y = datos[:,1]

    ##nº de ejemplos de entrenamiento
    m = len(X)
    ##"tamanio de los saltos que realizamos para acercarnos al minimo"
    alpha = 0.01
    #Inicializamos los valores de theta0 y theta1
    t0 = t1 = 0

    for _ in range(1500):     
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += (t0 + t1 * X[i]) - Y[i]
            sum_1 += ((t0 + t1 * X[i]) - Y[i]) * X[i]
        t0 = t0 - (alpha/m) * sum_0
        t1 = t1 - (alpha/m) * sum_1

    plt.figure()
    plt.plot(X,Y, "x",color='red')
    min_x = min(X)
    max_x = max(X)

    min_y = t0 + t1*min_x
    max_y = t0 + t1*max_x
    plt.plot([min_x,max_x], [min_y,max_y],color='blue', linewidth=2)
    plt.show()

def regresion_lineal_2():
    datos = carga_csv('ex1data2.csv')

    X = datos[:, :-1]
    np.shape(X)
    Y = datos[:, -1]
    np.shape(Y)

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    normaliza(X)
    normaliza(Y)

    #Añadimos una columna de 1's
    X = np.hstack([np.ones([m, 1]), X])

    alpha = 0.01

    Thetas, costes = descenso_gradiente(X, Y, alpha)

def coste(X,Y, Thetha):
    m = len(X)
    sum = ((Thetha[0] + Thetha[1]*X) - Y)**2
    sum = np.sum(sum)
    return (sum/(2*m))

def coste2(X, Y, Theta):
    H = np.dot(X, Theta)
    aux = (H - Y)**2
    return aux.sum() / (2 * len(X))

def make_data (t0_range , t1_range , X , Y ):
    step = 0.1
    t0 = np.arange(t0_range[0], t0_range[1],step)
    t1 = np.arange(t1_range[0], t1_range[1],step)

    t0,t1 = np.meshgrid(t0,t1)

    Coste = np.empty_like(t0)

    for ix, iy in np.ndindex(t0.shape):
        Coste[ix,iy] = coste(X,Y,[t0[ix,iy], t1[ix,iy]])
    return [t0,t1, Coste]

def me_quiero_morir(cosos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(cosos[0], cosos[1], cosos[2],cmap= plt.get_cmap('rainbow'))
    plt.show()

def me_quiero_morir_v2(cosos):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(cosos[0], cosos[1], cosos[2], np.logspace(-2,3,20),cmap= plt.get_cmap('rainbow'))
    plt.show()

def me_quiero_morir_3(cosos):
    #Hacer grafiquita de alpha (parte 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(cosos[0], cosos[1], cosos[2], np.logspace(-2,3,20),cmap= plt.get_cmap('rainbow'))
    plt.show()
    
def normaliza(matriz):
    #xi = (xi - ui) / si
    #ui = media ; si = desviación típica
    matriz_normal = np.empty_like(matriz)

    u = np.mean(matriz, axis=0)
    s = np.std(matriz, axis=0)
    print('Media: ' + str(u))
    print('Desv: ' + str(s))

    matriz_normal = (matriz - u) / s

    return [matriz_normal, u, s]

def descenso_gradiente(X, Y, alpha):
    Thetas = np.zeros((np.shape(X)[1],np.shape(X)[0]))
    
    #for _ in range(1500):
    Thetas = gradiente(X, Y, Thetas, alpha)

    Coste = coste2(X,Y,Thetas)

    #for ix, iy in np.ndindex(Thetas.shape):
    #    Coste[ix,iy] = coste2(X,Y,Thetas)

    return [Thetas, Coste]

def gradiente(X, Y, Theta, alpha):
    theta_nueva = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    aux = (H - Y)
    for i in range(n):
        aux_i = aux * X[:, i]
        theta_nueva -= (alpha / m) * aux_i.sum()
    return theta_nueva

#regresion_lineal_1()
#datos = carga_csv('ex1data2.csv')
#me_quiero_morir(make_data([-10,10],[-1,4],datos[:,0],datos[:,1]))
#print(normaliza(datos))
regresion_lineal_2()