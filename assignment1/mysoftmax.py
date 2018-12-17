import numpy as np

def loss_softmax(W,X,y,reg):
    #print('\n\nW;',W)
    scores=np.dot(X,W)
    #print('scores:',scores[0])
    exp_s=np.exp(scores)
    #print('exp_s:',exp_s[0])
    exp_sum=np.sum(exp_s,axis=1)
    #print('exp_sum:',exp_sum)
    loss=0.0
    dW=np.zeros(W.shape)
    num_classes=W.shape[1]
    for i in range(X.shape[0]):
        loss+=-np.log(exp_s[i,y[i]]/exp_sum[i])
        dW[:,y[i]]+=-X[i]
        for j in range(num_classes):
            dW[:,j]+=(exp_s[i,j]/exp_sum[i])*X[i]
        #print('p[{}] :{:.3f}'.format(i,p[i]))
    loss/=X.shape[0]
    loss+=reg*np.sum(np.square(W))
    dW/=X.shape[0]
    dW+=reg*W*2
    '''
    grad2=2*reg*W
    grad1=np.zeros(W.shape)
    for i in range(X.shape[0]):
        a=np.zeros(W.shape).T
        a[y[i]]==X[i]*exp_s[i,y[i]]*(1/exp_sum[i])*(1/p[i])*(-1/X.shape[0])
        grad1+=a.T

        b=np.zeros(W.shape).T
        for j in range(W.shape[-1]):
            b[j]=X[i]*exp_s[i,j]*(-exp_s[i,y[i]]/np.square(exp_sum[i]))*(1/p[i])*(-1/X.shape[0])
        grad1+=b.T

    grad=grad1+grad2
    '''
    return loss,dW


