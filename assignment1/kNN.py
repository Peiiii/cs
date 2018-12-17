import numpy as np
import pickle,time
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10 as load_CIFAR10
cifar10_dir=r'cs231n\datasets\cifar-10-batches-py'

class kNearestNeighbor(object):

    def __init__(self):
        pass
    def train(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict_labels(self,X,k=1):

        num_test=X.shape[0]
        y_pred=np.zeros(num_test,dtype='int64')
        dists=self.compute_distances_no_loops(X)
        #plt.imshow(dists,interpolation='none')
        #plt.show()
        #start=time.time()
        for i in range(num_test):
            index=np.argsort(dists[i])
            kInd=index[:k]
            closest_ys=self.y_train[kInd]
            y_pred[i]=np.argmax(np.bincount(closest_ys))
        #index=np.argsort(dists,axis=1)
        ##kInd=index[:,:k]
        #closest_ys=self.y_train[kInd]
        #for i in range(num_test):
         #   y_pred[i]=np.argmax((np.bincount(closest_ys[i])))
        #end=time.time()
        #print('Time consumed here:{:.3f}'.format(end-start))
        return y_pred
    def compute_distances_two_loops(self,X):
        #compute L2 distances
        num_train=self.X_train.shape[0]
        num_test=X.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j]=np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))
        return dists
    def compute_distances_one_loop(self,X):
        num_train=self.X_train.shape[0]
        num_test=X.shape[0]
        dists=np.zeros(num_test,num_train)
        for i in range(num_test):
            dists[i,:]=np.sqrt(np.sum(np.square(self.X_train-X[i]),axis=1))
        return dists
    def compute_distances_no_loops(self,X):

        dists=(-2)*np.dot(X,self.X_train.T)
        sq1=np.sum(np.square(self.X_train),axis=1)
        sq2=np.sum(np.square(X),axis=1,keepdims=True)
        dists=np.add(dists,sq1)
        dists=np.add(dists,sq2)

        return dists
    def get_accuracy(self,y_test,y_pred):
        num_test=y_test.shape[0]
        num_correct=0
        for i in range(num_test):
            if y_test[i]==y_pred[i]:
                num_correct+=1
        accuracy=num_correct/num_test
        return accuracy


    def get_random_data(self,dim=10,num_train=10,num_test=5):
        #create X_train,y_train,X,y randomly
        ####################################
        #X_train=np.random.randint(0,9,(num_train,dim))
        X_train=self.get_X_train_datasets()
        y_train=np.zeros(X_train.shape[0],dtype='int64')
        for i in range(X_train.shape[0]):
            y_train[i]=int(np.sum(X_train[i])/10)
        num_t1=int(num_test/2)
        if num_test%2==0:
            num_t2=num_t1
        else:
            num_t2=num_t1+1
        test1=np.random.randint(0,6,(num_t1,dim))
        test2=np.random.randint(4,10,(num_t2,dim))
        X_test=np.concatenate((test1,test2),axis=0)
        y_test=np.zeros(num_test,dtype='int64')
        for i in range(num_test):
            y_test[i]=int(np.sum(X_test[i])/10)
        return X_train,y_train,X_test,y_test
    def get_X_train_datasets(self,num_per_class=10,dim=10,num_classes=10):
        datasets=[]
        for i in range(num_classes):
            datasets.append([])
        b=[]
        for i in range(num_classes):
            b.append(1)
        run=0
        while True:
            if run<=60:
                a=np.random.randint(0,10,(dim))
            elif run%3==0:
                a=np.random.randint(0,4,(dim))
            elif run%3==1:
                a=np.random.randint(4,10,(dim))
            elif run%3==2:
                a=np.random.randint(9,10,(dim))

            run+=1
            ind=int(np.sum(a)/10)
            if b[ind]==1:
                datasets[ind].append(a)
                if len(datasets[ind])==num_per_class:
                    b[ind]=0
                    #print('{}'.format(b))
                    end=True
                    for i in range(num_classes):
                        if b[i]==1:
                            end=False
                            break
                    if end==True:
                        break
        datasets=np.reshape(datasets,(num_classes*num_per_class,dim))
        return datasets
    def show_random_image(self,X):
        #for i in range(X.shape[0]):
        i=np.random.randint(0,X.shape[0])
        plt.imshow(X[i].astype('uint8'))
        plt.axis('off')
        plt.show()
def get_accuracy(y_test,y_pred):
    num_test=y_test.shape[0]
    num_correct=0
    for i in range(num_test):
        if y_test[i]==y_pred[i]:
            num_correct+=1
    accuracy=num_correct/num_test
    return accuracy
def jiaochayanzheng(X_train,y_train,num_folds):
    classifier=kNearestNeighbor()
    X_train_folds=np.array_split(X_train,num_folds)
    y_train_folds=np.array_split(y_train,num_folds)
    k_choices=[1,2,3,5,8,9,10,11,12,15,20,30,40,50]
    k_to_accuracies={}
    for k in k_choices:
        k_to_accuracies[k]=np.zeros(num_folds)
    for fold in range(num_folds):
        temp_X=X_train_folds[:]
        temp_y=y_train_folds[:]
        X_test=temp_X.pop(fold)
        y_test=temp_y.pop(fold)
        temp_X=np.array([y for x in temp_X for y in x])
        temp_y=np.array([y for x in temp_y for y in x])
        classifier.train(temp_X,temp_y)
        for k in k_choices:
            y_pred=classifier.predict_labels(X_test,k)
            accuracy=classifier.get_accuracy(y_test,y_pred)
            k_to_accuracies[k][fold]=accuracy
    return k_to_accuracies
def get_datasets(num_train=1000,num_test=200):
    X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)
    X_train=X_train.reshape((X_train.shape[0],-1))
    X_train=X_train[:num_train]
    y_train=y_train[:num_train]
    X_test=X_test.reshape((X_test.shape[0],-1))
    X_test=X_test[:num_test]
    y_test=y_test[:num_test]
    return X_train,y_train,X_test,y_test





if __name__=="__main__":

    #####################################
    start=time.time()
    classifier=kNearestNeighbor()
    X_train,y_train,X_test,y_test=get_datasets()
    k_to_accuracies=jiaochayanzheng(X_train,y_train,num_folds=5)
    for k,accuracy in k_to_accuracies.items():
        print('k={}, accuracy={}'.format(k,accuracy))
    end=time.time()
    print('Time consumed: {:.2f}'.format(end-start))
    ###########################
    #this classifier does best when k has a value of 1

