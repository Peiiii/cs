import numpy as np
import time
import matplotlib.pyplot as plt
from mysoftmax import loss_softmax as loss_softmax
from mySVM import loss_SVM as loss_SVM
from cs231n.classifiers.softmax import softmax_loss_naive as softmax_loss_naive
from cs231n.data_utils import load_CIFAR10 as load_CIFAR10
cifar10_dir=r'cs231n\datasets\cifar-10-batches-py'

from kNN import get_datasets as get_datasets
from kNN import get_accuracy as get_accuacy


from cs231n.classifiers.linear_classifier import LinearClassifier as LinearClassifier


class Linear_classfier(object):
    # W: a numpy array with shape(dim,num_classes).
    # where a image has dim dimensions,and all images are of num_classes classes.

    # X : A numpy array with shape(num_test,dim).
    def __init__(self):
        self.W =None
        self.labels=None
        self.num_classes=None
    def predict(self,X,num_classes=None):
        if type(num_classes)!=type(None):
            self.num_classes=num_classes
        elif type(self.num_classes)==type(None):
            self.num_classes=10
        if type(self.W)==type(None):
            print('W has a value of none,W has not been trained.')
            print('W is now given a group of random values')
            self.W=np.random.randn(X.shape[1],self.num_classes)*0.0001
        elif X.shape[1]!=self.W.shape[0]:
            raise ValueError('Input size not matched. \
            Input image should has the same size with training images')
        scores=np.dot(X,self.W)
        y_pred=np.argmax(scores,axis=1)
        if type(self.labels)!=type(None):
            labels_pred=self.labels[y_pred]
        else:
            labels_pred=None
        return y_pred,labels_pred,scores
    def train(self,X,y,learning_rate=1.5e-4,reg=1e-2,num_iters=200,loss_kind=0,num_classes=None,labels=None,):
        if labels!=None:
            self.lables=labels
        if self.labels!=None:
            self.num_classes=self.labels
        elif num_classes!=None:
            self.num_classes=num_classes
        if self.num_classes==None:
            self.num_classes=np.max(y)+1
        loss_history=[]
        num_train,dim=X.shape
        if type(self.W)==type(None):
            self.W=np.random.randn(dim,self.num_classes)*0.0001
            print('W:,',self.W)
        for i in range(num_iters):
            new_learning_rate=self.get_learning_rate(learning_rate,i)
            loss,grad=self.loss(X,y,reg,kind=loss_kind)
            print(i,',','loss: {:.5f}'.format(loss))
            loss_history.append(loss)
            #print('srate:',-learning_rate*grad)
            self.W+=-new_learning_rate*grad
            #print('W:',self.W)
           # print('grad:',grad)
        return loss_history
    def loss(self,X,y,reg=1e-5,kind=0):
        if kind==0:
            return loss_SVM(self.W,X,y,reg)
        elif kind==1:
            return loss_softmax(self.W,X,y,reg)
        elif kind==2:
            return softmax_loss_naive(self.W,X,y,reg)
    def get_learning_rate(self,initial_rate,i,mode=0):
        ##################################
        #    4e-8 is thr best.
        #################################
        lr2=[1e-7,1e-7,1e-7]
        lr5=[1e-7,1e-7,1e-7]
        lr30=[1e-7,1e-7,1e-7]
        lr100=[1e-7,1e-7,1e-7]
        lr200=[1e-7,1e-7,1e-7]
        lr350=[1e-7,1e-7,1e-7]
        lr500=[1e-7,1e-7,1e-7]
        lrinf=[1e-7,1e-7,1e-7]
        #  best rate for softmax  :   1e-7
        ##################################
        m=1
        if m==0:
            lr=[1e-7,1e-8,1e-7]
            learning_rate=lr[mode]
        elif m==1:
            learning_rate=initial_rate
        elif m==2:
            if i<2:
                learning_rate=lr2[mode]
            elif i<5:
                learning_rate=lr5[mode]
            elif i<30:
                learning_rate=lr30[mode]
            elif i<100:
                learning_rate=lr100[mode]
            elif i<200:
                learning_rate=lr200[mode]
            elif i<350:
                learning_rate=lr350[mode]
            elif i<500:
                learning_rate=lr500[mode]
            else:
                learning_rate=lrinf[mode]
        return learning_rate
    def multi_train_test(self,X,y,X_test,y_test,learning_rate=1.5e-4, reg=1e-2,num_iters=200,loss_kind=0,num_classes=None,labels=None,):
        if labels!=None:
            self.lables=labels
        if self.labels!=None:
            self.num_classes=self.labels
        elif num_classes!=None:
            self.num_classes=num_classes
        if self.num_classes==None:
            self.num_classes=np.max(y)+1
        num_train,dim=X.shape
        if type(self.W)==type(None):
            self.W=np.random.randn(dim,self.num_classes)*0.0001
        ##############up is initialization    ##################
        y_pred,labels,scores=self.predict(X_test)
        ini_accuracy=get_accuacy(y_test,y_pred)
        ini_W=self.W.copy()
        multi_loss_history=[]
        multi_loss_change_history=[]
        accuacies=[]
        for mode in range(3):
            self.W=ini_W.copy()
            loss_history=[]
            for i in range(num_iters):
                new_learning_rate=self.get_learning_rate(learning_rate,i,mode=mode)
                loss,grad=self.loss(X,y,reg,kind=loss_kind)
                print('{}, loss: {:.5f}'.format(i,loss))
                loss_history.append(loss)
                self.W+=-new_learning_rate*grad
            y_pred,labels,scores=self.predict(X_test)
            accuracy=get_accuacy(y_test,y_pred)
            accuacies.append(accuracy)
            multi_loss_history.append(loss_history)

            loss_change_history=[]
            for i in range(len(loss_history)):
                if i!=0:
                    loss_change_history.append(loss_history[i]-loss_history[i-1])
            multi_loss_change_history.append(loss_change_history)

        accuacies.append(ini_accuracy)
        return multi_loss_history,multi_loss_change_history,accuacies
def run_download_classifier():
    start=time.time()
    classifier=LinearClassifier()
    X_train,y_train,X_test,y_test=get_datasets(num_train=100,num_test=3)
    loss_history=classifier.train(X_train,y_train,learning_rate=3e-4,reg=1e-5,num_iters=50)
    y_pred=classifier.predict(X_test)
    #print('W:',classifier.W)
   # print('scores:',scores)
    print('y_pred:{}'.format(y_pred),'\n\ny_test:{}'.format(y_test))
    for i in range(len(loss_history)):
        print(loss_history[i])
    accuracy=get_accuacy(y_test,y_pred)
    end=time.time()
    print('accuracy:{}'.format(accuracy))
    print('Time consumed:{:.2f}'.format(end-start))
def run_my_classifier(learning_rate=1e-7,reg=1e-5,num_iters=300,loss_kind=1,num_train=1000,num_test=200):
    start=time.time()
    classifier=Linear_classfier()
    X_train,y_train,X_test,y_test=get_datasets(num_train=num_train,num_test=num_test)
    y_random_pred,la,sc=classifier.predict(X_test)
    random_accuracy=get_accuacy(y_test,y_random_pred)

    #print('y_random_pred:',y_random_pred)
    loss_history=classifier.train(X_train,y_train,learning_rate=learning_rate,reg=reg,num_iters=num_iters,loss_kind=loss_kind)
    loss_history=np.array(loss_history)
    y_pred,labels,scores=classifier.predict(X_test)
    print('y_random_pred:',y_random_pred)
    print('y_pred:{}'.format(y_pred),'\n\ny_test:{}'.format(y_test))
    loss_change_history=[0]
    for i in range(len(loss_history)):
        if i!=0:
            loss_change_history.append(loss_history[i]-loss_history[i-1])
        print(i,'  :  {:.5f}'.format(loss_history[i]))
    #print(y_test,'*********',y_pred)
    accuracy=get_accuacy(y_test,y_pred)
    end=time.time()
    print('random_accuracy:',random_accuracy)
    print('accuracy:{}'.format(accuracy))
    print('accuracy improved:{:.2f}'.format(accuracy-random_accuracy))
    plt.plot(loss_history,label='single')
    plt.legend()
    print('Time consumed:{:.2f}'.format(end-start))
    plt.show()
def run_multi_train_test():
    start=time.time()
    classifier=Linear_classfier()
    X_train,y_train,X_test,y_test=get_datasets(num_train=1000,num_test=200)


    mlh,mlch,accuracies=classifier.multi_train_test(X_train,y_train,X_test,y_test,learning_rate=3e-7,reg=1e-2,num_iters=30,loss_kind=1)
    mlh=np.array(mlh)
    mlch=np.array(mlch)
    ini_accracy=accuracies.pop(-1)
    accuracies=np.array(accuracies)
    print('loss histiry:')
    for i in range(len(mlh[0])):
        print(i,' :  {:.3f}   {:.3f}   {:.3f}'.format(mlh[0,i],mlh[1,i],mlh[2,i]))

    print('ini_accuracy  : {:.2f}'.format(ini_accracy))
    print('accuracies    : ',accuracies)
    print('accuracy improved: ',['{:.2f}'.format(x) for x in accuracies-ini_accracy])
    plt.plot(mlh[0],label='line1',color='r')
    plt.plot(mlh[1],label='line2',color='blue')
    plt.plot(mlh[2],label='line3',color='green')
    end=time.time()
    print('Time consumed:{:.2f}'.format(end-start))
    plt.legend()
    plt.show()

    pass
if __name__=='__main__':
    run_my_classifier(learning_rate=1e-7,reg=1e-5,num_iters=200,loss_kind=1,num_train=2000,num_test=400)
    #run_multi_train_test()





