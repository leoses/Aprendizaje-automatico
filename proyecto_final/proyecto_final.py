import numpy as np
from numpy.lib.shape_base import column_stack
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from pandas.io.parsers import read_csv
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
            visualize_boundary(auxX, y, s, "C1.png")

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
    plt.savefig(texto1 + 'vs'+ texto2 + '.png')
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
        plt.savefig(str(df.columns[i]) + '.png')
        plt.show()
        plt.close()

dibuja_rel_columnas()