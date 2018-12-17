import numpy as np
import time,copy
from cs231n.data_utils import load_CIFAR10 as load_CIFAR10
cifar10_dir=r'cs231n\datasets\cifar-10-batches-py'
import matplotlib.pyplot as plt
from cs231n.classifiers.neural_net import TwoLayerNet as twolayernet
class TwoLayerNet:

    def __init__(self,input_size=3072,hidden_size=10,output_size=10,std=1e-4):
        self.params={}
        #self.params['W1']=np.random.randn(input_size,hidden_size)*std
        self.params['b1']=np.random.randn(hidden_size)
        #self.params['W2']=np.random.randn(hidden_size,output_size)*std
        self.params['b2']=np.random.randn(output_size)
        self.params['mu']=np.random.uniform(low=0.0,high=1.0,size=(hidden_size,))
        self.params['beta']=np.random.uniform(low=-1.0,high=-1.0,size=(hidden_size,))
        self.weight_initialization(input_size,hidden_size,output_size)
        self.p=0.5
    def predict(self,X):
        X*=self.p
        W1,b1,W2,b2=self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']
        mid=np.dot(X,W1)+b1
        bn=self.batch_normalization(mid)
        a1=np.maximum(0,bn)
        scores=np.dot(a1,W2)+b2
        y_pred=np.argmax(scores,axis=1)
        return y_pred
    def train(self,X_train,y_train,X_test,y_test,learning_rate=1e-6,learning_rate_decay=0.95,reg=1e-3,num_iters=100):
        self.learning_rate,self.reg,self.X,self.y=learning_rate,reg,X_train,y_train
        loss_history,Wloss_history,train_acc_history,test_acc_history,mu_history,belta_history=\
            [],[],[],[],[],[]
        cache_history,v_history,step_history=[],[],[]

        v_decay,cache_decay=0.9,0.99
        vW1,vb1,vW2,vb2=np.zeros(self.params['W1'].shape),np.zeros(self.params['b1'].shape),\
                        np.zeros(self.params['W2'].shape),np.zeros(self.params['b2'].shape)
        cache_W1,cache_W2,cache_b1,cache_b2=np.zeros(self.params['W1'].shape),np.zeros(self.params['W2'].shape),\
                                            np.zeros(self.params['b1'].shape),np.zeros(self.params['b2'].shape)
        y_test_pred=self.predict(X_test)
        y_train_pred=self.predict(X_train)
        accuracy_test=get_accuracy(y_test,y_test_pred)
        accuracy_train=get_accuracy(y_train,y_train_pred)
        train_acc_history.append(accuracy_train)
        test_acc_history.append(accuracy_test)


        for i in range(1,num_iters+1):
            loss,Wloss=self.forword()
            grads=self.backword()

            loss_history.append(loss)
            Wloss_history.append(Wloss)
            mu_history.append(self.params['mu'][0])
            belta_history.append(self.params['beta'][0])

            vW1=(v_decay*vW1-(1-v_decay)*grads['dW1'])
            #/(1-v_decay**i)
            vW2=(v_decay*vW2-(1-v_decay)*grads['dW2'])
            vb1=(v_decay*vb1-(1-v_decay)*grads['db1'])
            vb2=(v_decay*vb2-(1-v_decay)*grads['db2'])

            cache_W1=(cache_decay*cache_W1 + (1-cache_decay)*(grads['dW1']**2))
            #/(1-cache_decay**i)
            cache_W2=(cache_decay*cache_W2 + (1-cache_decay)*(grads['dW2']**2))
            cache_b1=(cache_decay*cache_b1 + (1-cache_decay)*(grads['db1']**2))
            cache_b2=(cache_decay*cache_b2 + (1-cache_decay)*(grads['db2']**2))

            aa=learning_rate*vW1/(np.sqrt(cache_W1)+1e-3)
            self.params['W1']+=aa
            self.params['b1']+=learning_rate*vb1/(np.sqrt(cache_b1)+1e-3)
            self.params['W2']+=learning_rate*vW2/(np.sqrt(cache_W2)+1e-3)
            self.params['b2']+=learning_rate*vb2/(np.sqrt(cache_b2)+1e-3)
            self.params['mu']+=-learning_rate*grads['dmu']
            #self.params['beta']+=-learning_rate*grads['dbeta']*5

            step_history.append(np.mean(aa))
            v_history.append(np.mean(vW1))
            cache_history.append(np.mean(cache_W1))
            learning_rate=self.learning_rate*np.exp(-0.01*i)
            if i%5==0:
                y_test_pred=self.predict(X_test)
                y_train_pred=self.predict(X_train)
                accuracy_test=get_accuracy(y_test,y_test_pred)
                accuracy_train=get_accuracy(y_train,y_train_pred)

                train_acc_history.append(accuracy_train)
                test_acc_history.append(accuracy_test)
        return [loss_history,Wloss_history,train_acc_history,test_acc_history,\
                mu_history,belta_history,cache_history,v_history,step_history]
    def forword(self):
        X=self.X
        y=self.y
        reg=self.reg
        n=X.shape[1]
        W1,b1,W2,b2,mu,beta=self.params['W1'],self.params['b1'],self.params['W2'],\
                             self.params['b2'],self.params['mu'],self.params['beta']
        mid=np.dot(X,W1)+b1
        ####bacth normalization####
        mid_means=np.mean(mid,axis=0)
        mid_vars=np.sum(np.square(mid-mid_means),axis=0)/mid.shape[0]
        bn0=(mid-mid_means)/np.sqrt(mid_vars)
        bn1=mu*bn0+beta
        ########batch normalization ends######
        p=0.5
        self.p=p
        a1=np.maximum(0,bn1)
        M1=np.random.rand(*a1.shape)<5
        a1*=M1
        scores=np.dot(a1,W2)+b2
        #M2=np.random.rand(*scores.shape)<p
        #scores*=M2

        y_pred=np.argmax(scores,axis=1)
        ###########################start of loss computing################
        num_input=X.shape[0]
        exp_scores=np.exp(scores)
        sum_exp_scores=np.sum(exp_scores,axis=1)
        loss=0.0
        exp_scores_norm=np.divide(exp_scores,np.matrix(sum_exp_scores).T)
        loss=np.sum(-np.log(exp_scores_norm[range(exp_scores_norm.shape[0]),y]))/num_input
        Wloss=reg*np.sum(np.square(W1))+reg*np.sum(np.square(W2))+reg*np.sum(np.square(b2))+reg*np.sum(np.square(b1))
        loss+=Wloss

        self.temp={'mid':mid,'bn0':bn0,'mid_means':mid_means,'mid_vars':mid_vars,\
                   'a1':a1,'exp_scores_norm':exp_scores_norm,'M1':M1}
        ###################end of loss computing####################
        return loss,Wloss
    def backword(self):
        X,y,reg,n,num_input=self.X,self.y,self.reg,self.X.shape[1],self.X.shape[0]
        W1,b1,W2,b2,mu,beta=self.params['W1'],self.params['b1'],self.params['W2'],\
                                               self.params['b2'],self.params['mu'],self.params['beta']
        mid,mid_means,mid_vars,bn0,a1,exp_scores_norm=self.temp['mid'],self.temp['mid_means'],self.temp['mid_vars'],\
                                                      self.temp['bn0'],self.temp['a1'],self.temp['exp_scores_norm']
        M1=self.temp['M1']
        ####s########tart of gradients computing###################
        dscores=exp_scores_norm
        dscores[range(exp_scores_norm.shape[0]),y]+=-1
        dscores/=num_input
        dscores=np.array(dscores)
        dW2=np.dot(a1.T,dscores)+reg*W2*2
        db2=np.sum(dscores,axis=0)+reg*b2*2
        db2=db2
        da1=np.dot(dscores,W2.T)
        da1[a1==0]=0
        #da1/=self.p
        da1=np.array(da1)
        M1=np.array(M1)
        da1*=M1

        ####bn gradients########
        dbn1=np.array(da1)##dbn
        dbn0=mu*dbn1
        dmu=np.sum(bn0*dbn1,axis=0)
        dbeta=np.sum(dbn1,axis=0)

        dmid=np.zeros(da1.shape)
        dmid[:,:]=-(mid[:,:]-mid_means[:])*np.sum( bn0[:,:]  ,axis=0)/(n*np.sqrt(mid_vars[:]))
        dmid+=-np.sum(dbn0,axis=0)/n + dbn0.reshape(dmid.shape)
        dmid[:,:]/=np.sqrt(mid_vars[:])
        #####bn gradients######


        dW1=np.dot(X.T,dmid)+reg*W1*2
        db1=np.sum(da1,axis=0)+reg*b1*2
        db1=db1
        grads={'dW2':np.array(dW2),'db2':np.array(db2),\
               'dW1':np.array(dW1),'db1':np.array(db1),'dmu':dmu,'dbeta':dbeta}
        ###########end of gradients computing##################
        return grads
    def weight_initialization(self,input_size,hidden_size,output_size):
        self.params['W1']=0.01*np.random.randn(input_size,hidden_size)/np.sqrt(input_size/2.0)
        self.params['W2']=0.01*np.random.randn(hidden_size,output_size)/np.sqrt(input_size/2.0)
    def batch_normalization(self,X):
        X_means=np.mean(X,axis=0)
        X_vars=np.sum(np.square(X-X_means),axis=0)/X.shape[0]
        bn0=(X-X_means)/np.sqrt(X_vars)
        bn1=bn0*self.params['mu']+self.params['beta']
        return bn1
    def run(self,X_train,y_train,X_test,y_test,num_iters=100,learning_rate=1,reg=3e-2):
        rvs=self.train(X_train,y_train,X_test,y_test,\
                       num_iters=num_iters,learning_rate=learning_rate,reg=reg)
        y_pred=self.predict(X_test)
        accuracy=get_accuracy(y_test,y_pred)
        return accuracy
    def loss(self,X,y,reg=1e-3):
        n=X.shape[1]
        W1,b1,W2,b2,mu,beta=self.params['W1'],self.params['b1'],self.params['W2'],\
                             self.params['b2'],self.params['mu'],self.params['beta']
        mid=np.dot(X,W1)+b1
        ####bacth normalization####
        mid_means=np.mean(mid,axis=0)
        mid_vars=np.sum(np.square(mid-mid_means),axis=0)/mid.shape[0]
        bn0=(mid-mid_means)/np.sqrt(mid_vars)
        bn1=mu*bn0+beta
        ########batch normalization ends######
        a1=np.maximum(0,bn1)
        scores=np.dot(a1,W2)+b2
        y_pred=np.argmax(scores,axis=1)
        ###########################start of loss computing################
        num_input=X.shape[0]
        exp_scores=np.exp(scores)
        sum_exp_scores=np.sum(exp_scores,axis=1)
        loss=0.0
        grads={}
        exp_scores_norm=np.divide(exp_scores,np.matrix(sum_exp_scores).T)
        loss=np.sum(-np.log(exp_scores_norm[range(exp_scores_norm.shape[0]),y]))/num_input
        Wloss=reg*np.sum(np.square(W1))+reg*np.sum(np.square(W2))+reg*np.sum(np.square(b2))+reg*np.sum(np.square(b1))
        loss+=Wloss
        ###################end of loss computing####################

        ####s########tart of gradients computing###################
        dscores=exp_scores_norm
        dscores[range(exp_scores_norm.shape[0]),y]+=-1
        dscores/=num_input
        dW2=np.dot(a1.T,dscores)+reg*W2*2
        db2=np.sum(dscores,axis=0)+reg*b2*2
        db2=db2.getA1()
        da1=np.dot(dscores,W2.T)
        da1[a1==0]=0

        ####bn gradients########
        dbn1=np.array(da1)##dbn
        dbn0=mu*dbn1
        dmu=np.sum(bn0*dbn1,axis=0)
        dbeta=np.sum(dbn1,axis=0)

        dmid=np.zeros(da1.shape)
        dmid[:,:]=-(mid[:,:]-mid_means[:])*np.sum( bn0[:,:]  ,axis=0)/(n*np.sqrt(mid_vars[:]))
        dmid+=-np.sum(dbn0,axis=0)/n + dbn0.reshape(dmid.shape)
        dmid[:,:]/=np.sqrt(mid_vars[:])
        #####bn gradients######


        dW1=np.dot(X.T,dmid)+reg*W1*2
        db1=np.sum(da1,axis=0)+reg*b1*2
        db1=db1.getA1()
        grads={'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1,'dmu':dmu,'dbeta':dbeta}
        ###########end of gradients computing##################
        return loss,grads,Wloss


def get_datasets(num_train=100,num_test=20,preprocessing=True):
    X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)
    X_train-=np.mean(np.mean(np.mean(X_train,axis=0),axis=0),axis=0)   ###data preprocessing
    X_train=X_train.reshape((X_train.shape[0],-1))
    X_train=X_train[:num_train]
    y_train=y_train[:num_train]
    X_test-=np.mean(np.mean(np.mean(X_test,axis=0),axis=0),axis=0)     ###data preprocessing
    X_test=X_test.reshape((X_test.shape[0],-1))
    X_test=X_test[:num_test]
    y_test=y_test[:num_test]
    return X_train,y_train,X_test,y_test
def get_accuracy(y_test,y_pred):
    num_test=y_test.shape[0]
    num_correct=0
    for i in range(num_test):
        if y_test[i]==y_pred[i]:
            num_correct+=1
    accuracy=num_correct/num_test
    return accuracy



def draw_history(histories=[],labels=[]):
    colors=['red','green','blue','yellow','gray']
    for i in range(len(histories)):
        plt.plot(histories[i],label=labels[i],color=colors[i])
    plt.legend()
    plt.show()


def run_my_net(num_iters=50):
    X_train,y_train,X_test,y_test=get_datasets(num_train=5000,num_test=2000)
    net=TwoLayerNet(hidden_size=300,input_size=3072,output_size=10)
    ini_y_pred=net.predict(X_test)
    rvs=net.train(X_train,y_train,X_test,y_test,num_iters=num_iters,learning_rate=10**(-0.5),reg=10**(-5))
    loss_history=rvs[0]
    Wloss_history=rvs[1]
    train_acc_history=rvs[2]
    test_acc_history=rvs[3]
    mu_history=rvs[4]
    beta_history=rvs[5]
    cache_history,v_history,step_history=rvs[6],rvs[7],rvs[8]
    for i in  range(len(loss_history)):
        print(i,' loss:   {:.4f}'.format(loss_history[i]),'Wloss:  {:.4f}'.format(Wloss_history[i]))
    y_pred=net.predict(X_test)
    accuracy=get_accuracy(y_test,y_pred)
    ini_accuracy=get_accuracy(y_test,ini_y_pred)
    print('ini_accuracy:      {:.3f}'.format(ini_accuracy))
    print('accuracy:          {:.3f}'.format(accuracy))
    print('accuracy improved: {:.3f}'.format(accuracy-ini_accuracy))

    draw_history([loss_history,Wloss_history],['loss_history','Wloss_history'])
    draw_history([mu_history,beta_history],['mu_history','beta_history'])
    draw_history([train_acc_history,test_acc_history],['train_acc_history','test_acc_history'])
    #draw_history([v_history],['v_history'])
    #draw_history([cache_history],['cache_history'])
    #draw_history([step_history],['step_history'])
def run_your_net(num_iters=300):
    X_train,y_train,X_test,y_test=get_datasets(num_train=2000,num_test=400)
    net=twolayernet(hidden_size=100,input_size=3072,output_size=10)
    ini_y_pred=net.predict(X_test)
    return_value=net.train(X_train,y_train,X_test,y_test,num_iters=num_iters,reg=1,learning_rate=1e-3)
    loss_history=return_value['loss_history']
    for i in  range(len(loss_history)):
        print(i,' loss:   {:.4f}'.format(loss_history[i]))
    #print(return_value['train_acc_history'],'\n',return_value['val_acc_history'])
    y_pred=net.predict(X_test)
    accuracy=get_accuracy(y_test,y_pred)
    ini_accuracy=get_accuracy(y_test,ini_y_pred)
    print('ini_accuracy:      {:.3f}'.format(ini_accuracy))
    print('accuracy:          {:.3f}'.format(accuracy))
    print('accuracy improved: {:.3f}'.format(accuracy-ini_accuracy))
    plt.plot(loss_history,color='red',label='loss history')
    plt.legend()
    plt.show()
    plt.plot(return_value['train_acc_history'],color='green',label='train_acc_history')
    plt.plot(return_value['val_acc_history'],color='blue',label='val_acc_history')
    plt.legend()
    plt.show()
def run_a_net(net,X_train,y_train,X_test,y_test,num_iters=200,learning_rate=1e-3,reg=5e-4,mod=0):
    ini_y_pred=net.predict(X_test)
    if mod==0:
        return_value=net.train(X_train,y_train,num_iters=num_iters,learning_rate=learning_rate,reg=reg)
    else:
        return_value=net.train(X_train,y_train,X_test,y_test,num_iters=num_iters,learning_rate=learning_rate,reg=reg)
    '''
    loss_history=return_value['loss_history']
    for i in  range(len(loss_history)):
        print(i,' loss:   {:.4f}'.format(loss_history[i]))
    print(return_value['train_acc_history'],'\n',return_value['val_acc_history'])
    '''
    y_pred=net.predict(X_test)
    accuracy=get_accuracy(y_test,y_pred)
    ini_accuracy=get_accuracy(y_test,ini_y_pred)
    print('ini_accuracy:      {:.3f}'.format(ini_accuracy))
    print('accuracy:          {:.3f}'.format(accuracy))
    print('accuracy improved: {:.3f}'.format(accuracy-ini_accuracy))
    return accuracy,return_value,y_pred,net.params['W1'][0,:10],X_test[0,:10]
    '''
    plt.plot(loss_history,color='red',label='loss history')
    plt.legend()
    plt.show()
    plt.plot(return_value['train_acc_history'],color='green',label='train_acc_history')
    plt.plot(return_value['val_acc_history'],color='blue',label='val_acc_history')
    plt.legend()
    plt.show()
    '''
def compare():
    start=time.time()
    num_iters=200
    net1=TwoLayerNet(hidden_size=100,input_size=3072,output_size=10)
    net2=TwoLayerNet(hidden_size=100,input_size=3072,output_size=10)
    net3=TwoLayerNet(hidden_size=100,input_size=3072,output_size=10)
    net2.params,net3.params=copy.deepcopy(net1.params),copy.deepcopy(net1.params)
    X_train,y_train,X_test,y_test=get_datasets(num_train=1000,num_test=200)
    ret1=run_a_net(net1,X_train,y_train,X_test,y_test,num_iters=200,reg=1,learning_rate=1e-3)
    end1=time.time()

    #X_train,y_train,X_test,y_test=get_datasets(num_train=1000,num_test=200)
    ret2=run_a_net(net2,X_train,y_train,X_test,y_test,num_iters=200,reg=1,learning_rate=1e-3)
    end2=time.time()

    #X_train,y_train,X_test,y_test=get_datasets(num_train=1000,num_test=200)
    ret3=run_a_net(net3,X_train,y_train,X_test,y_test,num_iters=200,reg=1,learning_rate=1e-3)
    end3=time.time()

    print(net1.params['W1'][0,:10],net2.params['W1'][0,:10])
    #print('y1:{}\ny2:{}'.format(ret1[2][:10],ret2[2][:10]))
    print('acc_1:{:.4f}\nacc_2:{:.4f}\nacc_3:{:.4f}'.format(ret1[0],ret2[0],ret3[0]))
    print('time1:{:.2f}\ntime2:{:.2f}\ntime3:{:2f}'.format(end1-start,end2-start,end3-start))

    plt.plot(ret1[1],color='r',label='net1')
    plt.plot(ret2[1],color='green',label='net2')
    plt.plot(ret3[1],color='blue',label='net3')
    plt.legend()
    plt.show()
def test_reg_lr():
    X_train,y_train,X_test,y_test=get_datasets(num_train=400,num_test=200)
    net=TwoLayerNet(hidden_size=100,input_size=3072,output_size=10)
    accuracies=[]
    lglrs=[]
    lgregs=[]
    lglr=-0.5
    numlr=1
    numreg=10
    for i in range(numlr):
        lgreg=-6
        lglrs.append(lglr)
        for j in range(numreg):
            if i==0:
                lgregs.append(lgreg)
            net_copy=copy.deepcopy(net)
            lr=10**lglr
            reg=10**lgreg
            accuracy=net_copy.run(X_train,y_train,X_test,y_test,num_iters=50,learning_rate=lr,reg=reg)
            accuracies.append(accuracy)
            print(i,'  ','lglr={:.2f}  lgreg={:.2f}  accuracy:  {:.2f}'.format(lglr,lgreg,accuracy))
            lgreg+=0.4
        lglr+=0.03
    accuracies=np.array(accuracies)
    accuracies=accuracies.reshape(numlr,numreg)
    print(accuracies)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(911, frameon=True, xticks=[], yticks=[])
    the_table=plt.table(cellText=accuracies,rowLabels=lglrs,colLabels=lgregs)
    the_table.set_fontsize(15)
    the_table.scale(1,3)
    plt.show()


if __name__=='__main__':
    run_my_net()
    #run_your_net()
    #compare()
    #run_a_net()
    #test_reg_lr()






