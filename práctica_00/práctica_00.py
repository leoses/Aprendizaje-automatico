import numpy as np
import scipy.integrate
import random

import time
import matplotlib.pyplot as plt

#funcion de prueba
def cuadrado(x):
    return x*x

#no iterativo
def integra_mc(fun, a, b, num_puntos=10000):
    puntos = np.linspace(a, b+1, b-a)
    max = np.max(fun(puntos))

    puntosX = (np.random.rand(num_puntos) * (b-a)) + a 
    puntosY = np.random.rand(num_puntos)  * max

    puntosFuncionY = fun(puntosX)

    elems = np.sum(puntosY < puntosFuncionY)

    return ((max * (b-a)) * (elems /num_puntos))

#iterativo
def integra_mc_2(fun, a, b, num_puntos=10000):
    i = a
    max = fun(a)
    for i in range(b): ##remember for num in nums if num < 0
        if(max < fun(i)): max = fun(i)
    
    i = 0
    puntosX = []
    puntosY = []
    for i in range(num_puntos):
        puntosX.append(random.random() * (b-a) + a)
        puntosY.append(random.random() * max)

    puntosFuncionY = []
    i = 0
    for i in range(num_puntos):
        puntosFuncionY.append(fun(puntosX[i]))
    
    cont = 0
    for i in range(num_puntos):
        if puntosY[i] < puntosFuncionY[i] : cont += 1
    
    return ((max * (b-a)) * (cont /num_puntos))


def pinta_grafica():
    masPunticos = 20000
    puntosTotales = 10000
    sizes = np.linspace(100,10000000,20)
    tiempo_bucle = []
    tiempo_np = []

    for size in sizes:
        x1 = random.randint(1,100)
        x2 = random.randint(1,100)

        if x1 > x2 : x1,x2 = x2,x1

        time_ini = time.process_time()
        integra_mc(cuadrado, x1, x2, puntosTotales)
        tiempo_np += [1000*(time.process_time() - time_ini)]
        
        time_ini = time.process_time()
        integra_mc_2(cuadrado, x1, x2, puntosTotales)
        tiempo_bucle += [1000*(time.process_time() - time_ini)]

        puntosTotales += masPunticos
    
    plt.figure()
    plt.scatter(sizes,tiempo_bucle,c = 'red',label='bucle')
    plt.scatter(sizes,tiempo_np,c = 'blue',label='numpy')
    plt.legend()
    plt.savefig('time.png')
    plt.show()

#area = integra_mc(cuadrado, 0, 20000)
#x1 = np.random.uniform(1,100)
#print(type(x1))

#print(area)
#area2 = scipy.integrate.quad(cuadrado, 0,20)
#print(area2)
#print(type(area))
pinta_grafica()