import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.shape_base import column_stack
import matplotlib.pyplot as plt
from numpy.random import normal
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from pandas.io.parsers import read_csv
from scipy.optimize import fmin_tnc, minimize
from sklearn.preprocessing import PolynomialFeatures

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
    #Sigma = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])
    #acurracyOpt = 0
    #sOpt = {}

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


def reg_logistica_graficas():
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


def reg_logistica():
    print("Estamos cargando")
    df = read_csv('diabetes-dataset.csv', header=0)

    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })

    a = df.to_numpy().astype(float)
    X = a[:,:-1]
    y = a[:,-1]
    print("Ya hemos cargado")
    

    #Ejemplos de entrenamiento
    entX = X[0:(int)(2000*0.6)]
    enty = y[0:(int)(2000*0.6)]
    #Ejemplos de validacion
    Xval = X[(int)(2000*0.6): (int)(2000*0.9)]
    yval = y[(int)(2000*0.6): (int)(2000*0.9)]
    #Ejemplos de test
    Xtest = X[(int)(2000*0.9):2000]
    ytest = y[(int)(2000*0.9):2000]

    # Anadimos columna de 1s
    auxX = np.hstack([np.ones([np.shape(entX)[0], 1]), entX])
            
    theta_opt = optimizacion(auxX,enty).ravel()
    
    Xval = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval])
    print("Porcentaje de ejemplos de validacion clasificados correctamente: " + str(calcula_porcentaje(Xval, yval, theta_opt)) + " %")

    Xtest = np.hstack([np.ones([np.shape(Xtest)[0], 1]), Xtest])
    print("Ejemplos de test clasificados correctamente: " + str(calcula_porcentaje(Xtest, ytest, theta_opt)) + " %")
    
    # Calculamos coste
    cost = coste_vec(theta_opt,auxX,enty)

    # Mostramos por pantalla para confirmar que el valor es el esperado
    print("Coste minimo con theta optimizada: " + str(cost))

def func_coste_reg(theta,X,Y, lamb):
    m = len(X)
    return coste_vec(theta, X, Y) + lamb/2*m * np.sum(theta*theta)

def gradiente_reg(theta, X,Y, lamb):
    m = len(X)
    return gradiente(theta, X, Y) + lamb/m * theta

def pintaFronteraCircular(theta,X,Y,poly, landa, texto1, texto2):
     # Creamos grafica
    fig = plt.figure()
    
    # Adding axes on the figure
    ax = fig.add_subplot(111)

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
    plt.scatter(X[ pos , 0 ] , X[ pos , 1 ] , marker='+' , c='k', label = "Diabético")
    # Dibuja los ejemplos negativos
    plt.scatter(X[ neg, 0 ] , X[ neg , 1 ] , marker='.' , c='orange', label = "No diabético")

     # Anadimos leyenda a la grafica y la posicionamos bien
    plt.legend(loc = 'upper right')

    # Anadimos el texto a los ejes (como en la grafica que aparece de ejemplo)
    ax.set_xlabel(texto1, fontsize=10)
    ax.set_ylabel(texto2, fontsize=10)

    # guardamos grafica
    plt.savefig("gráficas/Reg_Polinomica/"+ texto1 + texto2 + ".png")
    plt.show()
    plt.close()

def reg_logistica_reg_graficas_poly():
    print("Estamos cargando")
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

            # Apartado 2.1
            landa = 1
            poly = PolynomialFeatures(3) # Hasta la sexta potencia
            newX = poly.fit_transform(auxX)
            
            theta = np.zeros(np.shape(newX)[1])
            print("Valor de la funcion de coste regularizada "+ str(func_coste_reg(theta, newX,y, landa)))

            # Apartado 2.3
            result = fmin_tnc(func_coste_reg,theta , gradiente_reg , args =(newX, y, landa))

            pintaFronteraCircular(result[0], auxX, y, poly, landa, df.columns[i], df.columns[j])

def regresion_logistica_reg():
    print("Estamos cargando")
    df = read_csv('diabetes-dataset.csv', header=0)

    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })

    a = df.to_numpy().astype(float)
    X = a[:,:-1]
    y = a[:,-1]
    print("Ya hemos cargado")

    #Ejemplos de entrenamiento
    entX = X[0:(int)(2000*0.6)]
    enty = y[0:(int)(2000*0.6)]
    #Ejemplos de validacion
    Xval = X[(int)(2000*0.6): (int)(2000*0.9)]
    yval = y[(int)(2000*0.6): (int)(2000*0.9)]
    #Ejemplos de test
    Xtest = X[(int)(2000*0.9):2000]
    ytest = y[(int)(2000*0.9):2000]

    poly = PolynomialFeatures(3) # Hasta la tercera potencia
    newX = poly.fit_transform(entX)
    Xval = poly.fit_transform(Xval)
    Xtest = poly.fit_transform(Xtest)
    landas = np.array([ 0.00001,0.001,0.01 ,0.03, 0.1, 0.3, 1, 3])

    best_score_val =  0
    landa_score_val = landas[0]
    best_score_training = 0
    landa_score_training = landas[0]
    for landa in landas:
        theta = np.zeros(np.shape(newX)[1])

        print("Valor de la funcion de coste regularizada "+ str(func_coste_reg(theta, newX,enty, landa)))

        theta_opt = fmin_tnc(func_coste_reg,theta , gradiente_reg , args =(newX, enty, landa))[0]
        p_val = calcula_porcentaje(Xval, yval, theta_opt)
        p_training = calcula_porcentaje(Xtest, ytest, theta_opt)
        print("Porcentaje de ejemplos de validacion clasificados correctamente: " + str(p_val) + " %")
        print("Ejemplos de test clasificados correctamente: " + str(p_training) + " %")

        if p_val > best_score_val:
            best_score_val = p_val
            landa_score_val = landa

        if p_training > best_score_training:
            best_score_training = p_training
            landa_score_training = landa
    
    print("Mejores resultados para los ejemplos de validacion: landa = " + str(landa_score_val) + " con un porcentaje de acierto de " + str(best_score_val) + " %")
    print("Mejores resultados para los ejemplos de test: landa = " + str(landa_score_training) + " con un porcentaje de acierto de " + str(best_score_training) + " %")

def calcula_propagacion(X, theta1, theta2):
    m = np.shape(X)[0]
    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, theta1.T)
    A2 = np.hstack([np.ones([m, 1]), func_sigmoid(Z2)])
    Z3 = np.dot(A2, theta2.T)
    A3 = func_sigmoid(Z3)
    return A1, A2, A3

def gradiente_regularizacion(gradiente, m, reg, theta):
    columnaGuardada = gradiente[0]
    gradiente = gradiente + (reg/m)*theta
    gradiente[0] = columnaGuardada
    return gradiente

def coste_v2(X, y, t1, t2):
    m = np.shape(X)[0]
    a1, a2, h = calcula_propagacion(X, t1, t2)
    return (1/m)*np.sum(-y*np.log(h) - (1-y)*np.log(1-h))

def coste_reg_v2(X, y, t1, t2, reg):
    m = np.shape(X)[0]
    #En el coste reg no hay que incluir la primera columna de t1 y t2
    otra = (reg/(2*m)) * (np.sum(np.power(t1[1:], 2)) + np.sum(np.power(t2[1:], 2)))
    return coste_v2(X, y, t1, t2) + otra

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
        a1t = A1[t, :] 
        a2t = A2[t, :] 
        ht = H[t, :] 
        yt = Y[t]
 
        d3t = ht - yt 
        d2t = np.dot(Theta2.T, d3t)*(a2t*(1-a2t)) 
 
        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
 

    gradiente1 = Delta1/m
    gradiente2 = Delta2/m

    gradiente1 = gradiente_regularizacion(gradiente1, m, reg, Theta1)
    gradiente2 = gradiente_regularizacion(gradiente2, m, reg, Theta2)
 
    return coste_reg_v2(X, Y, Theta1, Theta2, reg), np.concatenate([np.ravel(gradiente1), np.ravel(gradiente2)])

def y_oneHot(y, numLabels):
    m = len(y)
 
    y_onehot = np.zeros((m, numLabels))
 
    for i in range(m):
        y_onehot[i][y[i]] = 1
 
    return y_onehot

def optimize_backprop_and_check (t1, t2,num_entradas, num_ocultas, num_etiquetas, reg, X, y_onehot, laps):
    
    eini = 0.77
    params = np.concatenate([np.ravel(t1),np.ravel(t2)])
    pesos = np.random.uniform(-eini,eini, params.shape[0])
 
    out = minimize(fun = backprop, x0 = pesos, args = (num_entradas, num_ocultas, num_etiquetas, X, y_onehot, reg), method='TNC', jac = True, options = {'maxiter': laps})
 
    Thetas1 = out.x[:(num_ocultas*(num_entradas+1))].reshape(num_ocultas,(num_entradas+1))
    Thetas2 = out.x[(num_ocultas*(num_entradas+1)):].reshape(num_etiquetas,(num_ocultas+1))
   
    return Thetas1, Thetas2
    


def redes_neuronales():
    print("Estamos cargando")
    df = read_csv('diabetes-dataset.csv', header=0)

    for i in range(1, len(df.columns) - 1):
        df[df.columns[i]] = df[df.columns[i]].replace({ 0 : df[df.columns[i]].mean() })

    a = df.to_numpy().astype(float)
    b = df.to_numpy().astype(int)
    X = a[:,:-1]
    y = b[:,-1]
    print("Ya hemos cargado")

    #Ejemplos de entrenamiento
    entX = X[0:(int)(2000*0.6)]
    enty = y[0:(int)(2000*0.6)]
    #Ejemplos de validacion
    Xval = X[(int)(2000*0.6): (int)(2000*0.9)]
    yval = y[(int)(2000*0.6): (int)(2000*0.9)]
    #Ejemplos de test
    Xtest = X[(int)(2000*0.9):2000]
    ytest = y[(int)(2000*0.9):2000]

    num_etiquetas = 2
    auxY = y_oneHot(enty, num_etiquetas)
    num_ocultas = 1000
    num_entradas = np.shape(entX)[1]
    
    landas = np.array([ 0.00001,0.001,0.01 ,0.03, 0.1, 0.3, 1, 3])

    theta1= np.zeros((num_ocultas, num_entradas+1))
    theta2 = np.zeros((num_etiquetas, num_ocultas+1))

    for landa in landas:    
        T1, T2 = optimize_backprop_and_check(theta1, theta2,num_entradas,num_ocultas, num_etiquetas, landa,entX, auxY, 70)

        print("Para landa = " + str(landa))
        A1,A2,H = calcula_propagacion(Xval,T1, T2)
        maxIndices = np.argmax(H,axis=1) 
        acertados = np.sum(maxIndices==yval.ravel())
        print("Porcentaje acertados validacion: " + str(acertados*100/np.shape(H)[0]) + "%")

        A1,A2,H = calcula_propagacion(Xtest,T1, T2)
        maxIndices = np.argmax(H,axis=1) 
        acertados = np.sum(maxIndices==ytest.ravel())
        print("Porcentaje acertados test: " + str(acertados*100/np.shape(H)[0]) + "%")


redes_neuronales()