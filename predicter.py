import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(0)
pd.set_option('display.max_columns', 500)

#####################size is 1x13#####################
heart=pd.read_csv('heart_disease_dataset.csv',sep=';')

#data prep
target=heart['target']
X=heart.drop('target',axis=1)
X=np.array(X)
X.reshape((1,-1))
Y=np.array(target).reshape((-1,1))
standart=StandardScaler().fit_transform(X)# transform to Z distrubution


x_train,x_test,y_train,y_test=train_test_split(standart,Y,random_state=12,test_size=0.5)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)

iterations=200
weight_0_1=np.random.random((X.shape[1],4))-0.5
weight_1_2=np.random.random((4,6))-0.5
weight_2_3=np.random.random(((6,1)))-0.5


for iterat in range(100):
    all_error=0
    for i in range(len(x_train)):
        layer_0=x_train[i:i+1]
        target_is=y_train[i:1+i]
        layer_1=sigmoid(np.dot(layer_0,weight_0_1))
        layer_2=sigmoid(np.dot(layer_1,weight_1_2))
        layer_out=sigmoid(np.dot(layer_2,weight_2_3))


        error=np.sum((layer_out-target_is)**2)
        all_error+=error

        ### BACK PROPOGATION ###
        layer_out_delta=(target_is-layer_out)*sigmoid_derivative(layer_out)
        layer_2_delta=layer_out_delta.dot(weight_2_3.T)*sigmoid_derivative(layer_2)
        layer_1_deta=layer_2_delta.dot(weight_1_2.T)*sigmoid_derivative(layer_1)
        
        ### MAKE WEIGHT CORRECTION ###
        weight_2_3+=layer_2.T.dot(layer_out_delta)
        weight_1_2+=layer_1.T.dot(layer_2_delta)
        weight_0_1+=layer_0.T.dot(layer_1_deta)

        print(target_is,'test target')
        print(layer_out,'test predict')
        print(error,'ERROR')
    ### CHECK ###
    layer_0 = x_test[i:i + 1]
    target_is = y_test[i:1 + i]
    layer_1 = sigmoid(np.dot(layer_0, weight_0_1))
    layer_2 = sigmoid(np.dot(layer_1, weight_1_2))
    layer_out = sigmoid(np.dot(layer_2, weight_2_3))
    print(target_is,'target test')
    print(layer_out,'predicted')
    print(all_error,'ALL ERROR')




