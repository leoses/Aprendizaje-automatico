import numpy as np
from numpy.lib.shape_base import column_stack
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from pandas.io.parsers import read_csv
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures
import codecs

def visualize_boundary(X, y, svm, file_name):
    x1 = np.linspace(X[:, 0].min()-0.01 , X[:, 0].max()+0.01 , 100)
    x2 = np.linspace(X[:, 1].min()-0.01, X[:, 1].max()+0.01 , 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.show()
    plt.close()

def carga_csv(s):
    valores = read_csv (s , header=1).to_numpy()
    return valores.astype(float)

def apartado1_1():
    df = read_csv('diabetes-dataset.csv', header=0)

    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })

    a = df.to_numpy().astype(float)
    X = a[:,:-1]
    y = a[:,-1]
    print("Ya hemos cargado")

    for i in range(1, np.shape(X)[1]):
        for j in range(i+1, np.shape(X)[1]):
            auxX = column_stack((X[:,i], X[:,j]))
            s = svm.SVC(kernel='linear', C=1.0)
            s.fit(auxX,y)
            visualize_boundary(auxX, y, s, "gráficas/Rectas_svm/" + df.columns[i] + df.columns[j] + ".png")

def apartado2():
    data = carga_csv('diabetes-dataset.csv')
    
    auxy = data[:,-1]
    auxX = data[:, :-1]

    x = auxX[0:(int)(2000*0.6)]
    y = auxy[0:(int)(2000*0.6)]
    Xval = auxX[(int)(2000*0.6): (int)(2000*0.9)]
    yval = auxy[(int)(2000*0.6): (int)(2000*0.9)]
    Xtest = auxX[(int)(2000*0.9):2000]
    ytest = auxy[(int)(2000*0.9):2000]

    C = np.array([ 0.01 ,0.03, 0.1, 0.3, 1, 3, 10,30])
    Sigma = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])
    acurracyOpt = 0
    sOpt = {}

    for c in C: 
        s = svm.SVC(kernel='linear', C = c)    
        s.fit(x, y)
        print("Acurracy" + str(accuracy_score(yval, s.predict(Xval))))
        print("Aciertos de clasificador: "+ str(accuracy_score(ytest,s.predict(Xtest))*100) + "%")

# Apartado 1.1, pintado de la grafica con los datos recibidos
def grafica_v1(X, Y, texto1, texto2):
    # creamos figura
    fig = plt.figure()
    # Adding axes on the figure
    ax = fig.add_subplot(111)
    
    # Obtiene un vector con los índices de los ejemplos positivos (1 en reg logistica)
    pos = np.where(Y == 1)
    # Obtener vector con los indices de los ejemplos negativos (0 en reg logistica)
    neg = np.where(Y == 0)
    
    # Dibuja los ejemplos positivos
    plt.scatter(X[pos , 0], X[pos , 1] , marker='+' , c='k', label = "Diabético")
    # Dibuja los ejemplos negativos
    plt.scatter(X[neg, 0], X[neg , 1] , marker='.' , c='orange', label = "No diabético")

    # Anadimos leyenda a la grafica y la posicionamos bien
    plt.legend(loc = 'upper right')

    # Anadimos el texto a los ejes (como en la grafica que aparece de ejemplo)
    ax.set_xlabel(texto1, fontsize=10)
    ax.set_ylabel(texto2, fontsize=10)

    # Guardamos en formato png
    plt.savefig('gráficas/Punticos/' + texto1 + texto2 + '.png')
    # Mostramos por pantalla
    plt.show()
    # Cerramos
    plt.close()

def dibuja_rel_columnas():
    df = read_csv('diabetes-dataset.csv', header=0)
 
    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })
 
    a = df.to_numpy().astype(float)
 
    X = a[:,:-1]
    y = a[:,-1]
    print(np.shape(X))
 
    for i in range(0,np.shape(X)[1]):
        for j in range(i+1, np.shape(X)[1]):
            auxX = column_stack((X[:,i], X[:,j]))
            grafica_v1(auxX, y, df.columns[i], df.columns[j])


def dibuja_histogramas():
    df = read_csv('diabetes-dataset.csv', header=0)
 
    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })
 
    for i,col in enumerate(df.drop(['Outcome'],axis=1)):
        fig = plt.figure()
        # Adding axes on the figure
        ax = fig.add_subplot(111)
        plt.hist(df[col][df['Outcome'] ==0], bins=20, fc=(1, 0, 1.0, 0.7), label="No Diabético")
        plt.hist(df[col][df['Outcome'] ==1], bins=20,fc=(1.0, 0.64, 0, 0.7) , label="Diabético")
        plt.legend()
        ax.set_xlabel( str(df.columns[i]), fontsize=12)
        ax.set_ylabel("Patients", fontsize=12)
        plt.savefig('gráficas/Histogramas/' + str(df.columns[i]) + '.png')
        plt.show()
        plt.close()

# Regresion logistica
# Aplicamos formula de la funcion sigmoide tal cual
# np.exp hace e^X siendo x lo que le pases como parametro
def func_sigmoid(X):
    return (1 / (1 + np.exp(-X)))

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

def func_coste_reg(theta,X,Y, lamb):
    m = len(X)
    return coste_vec(theta, X, Y) + lamb/2*m * np.sum(theta*theta)

def gradiente_reg(theta, X,Y, lamb):
    m = len(X)
    return gradiente(theta, X, Y) + lamb/m * theta

# Apartado 1.4, calculo del valor optimo de parametros
def optimizacion(X,Y):
    n=np.shape(X)[1]
    # Inicializamos todos los valores de theta a 0  
    theta=np.zeros(n)
    result = fmin_tnc(coste_vec,theta , gradiente , args =(X, Y))
    return result[0]

def pinta_frontera(X,Y,theta,texto1,texto2):
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
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    # Obtiene un vector con los índices de los ejemplos positivos (1 en reg logistica)
    pos=np.where(Y == 1)
    # Obtiene un vector con los índices de los ejemplos negativos (0 en reg logistica)
    neg=np.where(Y == 0)

    # Dibuja los ejemplos positivos
    plt.scatter(X[ pos , 1] , X[ pos , 2] , marker='+' , c='k', label = "Diabético")
    # Dibuja los ejemplos negativos
    plt.scatter(X[ neg, 1] , X[ neg , 2] , marker='.' , c='orange', label = "No diabético")

    # Anadimos leyenda a la grafica y la posicionamos bien
    plt.legend(loc = 'upper right')

    # Anadimos el texto a los ejes (como en la grafica que aparece de ejemplo)
    ax.set_xlabel(texto1, fontsize=10)
    ax.set_ylabel(texto2, fontsize=10)

    # guardamos grafica
    plt.savefig("gráficas/Rectas_Regresion/frontera" + texto1 + texto2 + ".png")
    # mostramos grafica
    plt.show()
    # Cerramos
    plt.close()

def pinta_frontera_circular(X,Y,theta,poly,landa,texto1,texto2):
    plt.figure()

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    
    h = func_sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
    xx2.ravel()]).dot(theta))
    
    h = h.reshape(xx1.shape)
    
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    # Obtiene un vector con los índices de los ejemplos positivos (1 en reg logistica)
    pos=np.where(Y == 1)
    # Obtiene un vector con los índices de los ejemplos negativos (0 en reg logistica)
    neg=np.where(Y == 0)

    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label = "Admited")
    # Dibuja los ejemplos negativos
    plt.scatter(X[neg, 0], X[neg, 1], marker='.', c='orange', label = "Not Admitted")

    plt.savefig("boundaryCircular" + str(landa) + ".png")
    plt.show()
    plt.close()

def apartado3():
    print("Estamos cargando")
    df = read_csv('diabetes-dataset.csv', header=0)

    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })

    a = df.to_numpy().astype(float)
    X = a[:,:-1]
    y = a[:,-1]
    print("Ya hemos cargado")

    # Calculo de theta optima para minimizar funcion de coste
    for i in range(1, np.shape(X)[1]):
        for j in range(i+1, np.shape(X)[1]):
            auxX = column_stack((X[:,i], X[:,j]))

            # Anadimos columna de 1s
            auxX = np.hstack([np.ones([np.shape(auxX)[0], 1]), auxX])
            n=np.shape(auxX)[1]
            
            theta_opt = optimizacion(auxX,y)
 
            # Calculamos coste
            cost = coste_vec(theta_opt,auxX,y)
            # Mostramos por pantalla para confirmar que el valor es el esperado
            print("Coste minimo con theta optimizada: " + str(cost))
 
            # Pintamos la frontera
            pinta_frontera(auxX,y,theta_opt,df.columns[i], df.columns[j])

def apartado4():
    print("Estamos cargando")
    df = read_csv('diabetes-dataset.csv', header=0)

    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })

    a = df.to_numpy().astype(float)
    X = a[:,:-1]
    y = a[:,-1]
    print("Ya hemos cargado")

    landa = 100
    poly = PolynomialFeatures(6) # Hasta la sexta potencia

    # Calculo de theta optima para minimizar funcion de coste
    for i in range(1, np.shape(X)[1]):
        for j in range(i+1, np.shape(X)[1]):
            auxX = column_stack((X[:,i], X[:,j]))
            newX = poly.fit_transform(auxX)
 
            theta = np.zeros(np.shape(newX)[1])
            print("Valor de la funcion de coste regularizada " + str(func_coste_reg(theta, newX, y, landa)))
 
            #result = fmin_tnc(func_coste_reg, theta, gradiente_reg, args = (newX, y, landa))

            # Pintamos la frontera
            pinta_frontera_circular(auxX, y, theta, poly, landa, df.columns[i], df.columns[j])

apartado3()