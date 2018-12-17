import numpy as np

def loss_SVM(W,X,y,reg):
    scores=np.dot(X,W)
    l2=reg*np.sum(np.square(W))
    print('reguration loss:',l2)
    dW=np.zeros(W.shape)
    num_classes=W.shape[1]
    num_train=X.shape[0]
    loss=0.0
    for i in range(num_train):
        for j in range(num_classes):
            margin=scores[i,j]-scores[i,y[i]]+1
            if j==y[i]:
                continue
            if margin>0:
                loss+=margin
                dW[:,j]+=X[i]
                dW[:,y[i]]+=-X[i]
    loss=loss/X.shape[0]
    dW=dW/X.shape[0]
    loss+=reg*np.sum(np.square(W))
    dW+=reg*W*2
    return loss,dW



