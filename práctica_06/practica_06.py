import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from process_email import preProcess, email2TokenList
from get_vocab_dict import getVocabDict
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

def apartado1_1():
    data = loadmat('ex6data1.mat')
    y = data['y']
    X = data['X']

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
    vocab = getVocabDict()

    #Leemos los datos y los convertimos para procesarlos
    spam= genera_0_1(vocab,"spam", 500)
    #todo 1s porque son spam todos
    y_Spam = np.ones(len(spam))

    hardHam = genera_0_1(vocab,"hard_ham", 250)
    y_HardHam = np.zeros(len(hardHam))

    easyHam= genera_0_1(vocab,"easy_ham", 2551)
    y_EasyHam = np.zeros(len(easyHam))
    
    x, y = genera_matrices(spam[:int(500*0.6)], hardHam[:int(250*0.6)], easyHam[:int(2551*0.6)],
     y_Spam[:int(500*0.6)], y_HardHam[:int(250*0.6)], y_EasyHam[:int(2551*0.6)])


    Xval, yval = genera_matrices(spam[int(500*0.6):int(500*0.9)], hardHam[int(250*0.6):int(250*0.9)], easyHam[int(2551*0.6):int(2551*0.9)],
     y_Spam[int(500*0.6):int(500*0.9)], y_HardHam[int(250*0.6):int(250*0.9)], y_EasyHam[int(2551*0.6):int(2551*0.9)])


    Xtest, ytest = genera_matrices(spam[int(500*0.9):], hardHam[int(250*0.9):], easyHam[int(2551*0.9):],
     y_Spam[int(500*0.9):], y_HardHam[int(250*0.9):], y_EasyHam[int(2551*0.9):])


    C = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])
    Sigma = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])
    acurracyOpt = 0
    sOpt = {}

    for c in C: 
        for sigma in Sigma: 
            s = svm.SVC(kernel='rbf' , C= c, gamma=1 / ( 2 * sigma **2) )    
            s.fit(x, y)
            accuracy = accuracy_score(yval, s.predict(Xval))
            if accuracy > acurracyOpt:
                acurracyOpt = accuracy 
                sOpt = s
    
    print("Aciertos de spam: "+ str(accuracy_score(ytest,sOpt.predict(Xtest))*100) + "%")

apartado2()