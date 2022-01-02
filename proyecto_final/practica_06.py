import numpy as np
from numpy.lib.shape_base import column_stack
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from pandas.io.parsers import read_csv
from process_email import preProcess, email2TokenList
from get_vocab_dict import getVocabDict
import codecs
import seaborn as sns
#sns.set()

def visualize_boundary(X, y, file_name):
    x1 = np.linspace(X[:, 0].min()-10 , X[:, 0].max()+10 , 100)
    x2 = np.linspace(X[:, 1].min()-10, X[:, 1].max()+10 , 100)
    x1, x2 = np.meshgrid(x1, x2)
    ##yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    #plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.show()
    plt.close()

def carga_csv(s):
    valores = read_csv (s , header=None).to_numpy()
    return valores.astype(float)

def apartado1_1():
    #data = loadmat('ex6data1.mat')
    #y = data['y']
    #X = data['X']
    print("Estamos cargando")
    data = carga_csv('diabetes-dataset-chikito.csv')
    print("Ya hemos cargado")
    y = data[:,-1]
    X = data[:, :2]

    s = svm.SVC( kernel='linear' , C=1.0)
    s.fit(X,y)

    s2 = svm.SVC(kernel='linear' , C=100.0)
    s2.fit(X,y.ravel())

    visualize_boundary(X,y,s,"C1.png")
    visualize_boundary(X,y,s2,"C100.png")

def apartado1_2(C, sigma):
    data = loadmat('ex6data2.mat')
    y = data['y']
    X = data['X']

    s = svm.SVC(kernel='rbf', C=C, gamma=1 / (2 * sigma**2)) 
    s.fit(X,y.ravel())

    visualize_boundary(X,y,s,"apartado2.png")


def apartado1_3():
    data = loadmat('ex6data3.mat')
    y = data['y']
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    C = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])
    Sigma = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])

    acurracyOpt = 0
    sOpt = 0
    
    for c in C:
        for sigma in Sigma:
            s = svm.SVC(kernel='rbf',C=c,gamma=1/(2 * sigma**2))
            s.fit(X,y.ravel())
            accuracy = accuracy_score(yval, s.predict(Xval))
            if(accuracy > acurracyOpt):
                sOpt = s
                acurracyOpt = accuracy 

    visualize_boundary(X,y,sOpt,"apartado3.png")

def procesa_email(vocab, email):
    mailVector =  np.zeros([len(vocab)])
    for word in email:
        if word in vocab:
            mailVector[vocab[word]-1] = 1
            
    return mailVector

def genera_0_1(vocab,folderName, size):
    emails = np.empty((size, len(vocab)))
    for i in range(1,size+1):
        print(i)
        contenido_email = codecs.open('{0}/{1:04d}.txt'.format(folderName, i), 'r',encoding='utf-8', errors='ignore').read()
        email = email2TokenList(contenido_email)
        email_procesado = procesa_email(vocab, email)
        emails[i-1] = email_procesado
    return emails

def genera_matrices(spam, hardHam, easyHam, y_spam, y_hardham, y_easyham):
    X = spam
    y = y_spam
    X = np.append(X,hardHam, axis=0)
    X = np.append(X,easyHam, axis=0)
    y = np.append(y,y_hardham)
    y = np.append(y,y_easyham)

    return X , y
def apartado2():
    ##vocab = getVocabDict()
##
    ###Leemos los datos y los convertimos para procesarlos
    ##spam= genera_0_1(vocab,"spam", 500)
    ###todo 1s porque son spam todos
    ##y_Spam = np.ones(len(spam))
##
    ##hardHam = genera_0_1(vocab,"hard_ham", 250)
    ##y_HardHam = np.zeros(len(hardHam))
##
    ##easyHam= genera_0_1(vocab,"easy_ham", 2551)
    ##y_EasyHam = np.zeros(len(easyHam))
    ##
    ##x, y = genera_matrices(spam[:int(500*0.6)], hardHam[:int(250*0.6)], easyHam[:int(2551*0.6)],
    ## y_Spam[:int(500*0.6)], y_HardHam[:int(250*0.6)], y_EasyHam[:int(2551*0.6)])
##
##
    ##Xval, yval = genera_matrices(spam[int(500*0.6):int(500*0.9)], hardHam[int(250*0.6):int(250*0.9)], easyHam[int(2551*0.6):int(2551*0.9)],
    ## y_Spam[int(500*0.6):int(500*0.9)], y_HardHam[int(250*0.6):int(250*0.9)], y_EasyHam[int(2551*0.6):int(2551*0.9)])
##
##
    ##Xtest, ytest = genera_matrices(spam[int(500*0.9):], hardHam[int(250*0.9):], easyHam[int(2551*0.9):],
    ## y_Spam[int(500*0.9):], y_HardHam[int(250*0.9):], y_EasyHam[int(2551*0.9):])
##
    print("Estamos cargando")
    data = carga_csv('diabetes-dataset.csv')
    print("Ya hemos cargado")
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
        s = svm.SVC(kernel='linear' , C= c)    
        s.fit(x, y)
        print("Acurracy" + str(accuracy_score(yval, s.predict(Xval))))
        print("Aciertos de clasificador: "+ str(accuracy_score(ytest,s.predict(Xtest))*100) + "%")

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

def prueba():
    df = read_csv('diabetes-dataset.csv', header=1)

    a = df.to_numpy().astype(float)
    ##for i,col in (df.drop(['Outcome'],axis=1)):
    ##    plt.figure()
    ##    sns.distplot(df[col])
    ##    plt.show()

    X = a[:,:-1]
    y = a[:,-1]
    print(np.shape(X))

    for i in range(0,np.shape(X)[1]):
        for j in range(i+1, np.shape(X)[1]):
            auxX = column_stack((X[:,i], X[:,j]))
            grafica_v1(auxX, y)

    ##for i,col in enumerate(df.drop(['Pregnancies','Outcome'],axis=1)):
    ##    plt.figure()
    ##    ax1=sns.distplot(df[col][df['Outcome']==1],label='Positive')
    ##    sns.distplot(df[col][df['Outcome']==0],label='Negative',ax=ax1)
    ##    plt.legend()
    ##    plt.show()

prueba()