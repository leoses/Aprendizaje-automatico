import numpy as np
import scipy.integrate

#funcion de prueba
def cuadrado(x):
    return x*x

#no iterativo
def integra_mc(fun, a, b, num_puntos=10000):
    puntos =np.linspace(a, b+1, b-a)
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
        i = i + 1

    

    
    
    return 0

area = integra_mc(cuadrado, 10, 20 )
print(area)
area2 = scipy.integrate.quad(cuadrado, 10,20)
print(area2)