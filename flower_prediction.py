#This model works on Iris Dataset 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def solve(mean,sd,l):
    ans=np.zeros(4)
    for i in range(4):
      ans[i]=0.399*np.exp(-(((l[i]-mean[i])/sd[i])**2)/2)/sd[i]
    return(ans)
def naive(l):
    data1=pd.DataFrame()
    if(len(l)!=4):
        return(0)
    else:
        mean=np.zeros(4)
        sd=np.zeros(4)
        result=np.zeros(3)
        class_label=train[train.columns[4]].unique()
        col=data.columns
        for i in range(3):
            for j in range(4):
                mean[j]=train[col[j]][train[train.columns[4]]==class_label[i]].mean()
                sd[j]=train[col[j]][train[train.columns[4]]==class_label[i]].std()
            prob=solve(mean,sd,l)
            result[i]=prob[0]*prob[1]*prob[2]*prob[3]
        sol=max(result)
        for i in range(len(result)):
            if(sol==result[i]):
               sol=i
               break
        return(class_label[i])
def pred():
    t=test.values
    sol=[]
    for i in range(len(t)):
        sol.append(naive(t[i]))
    return(np.array(sol).T)
def error():
    prediction=pred()
    count=0
    for i in range(len(prediction)):
        if(prediction[i]!=y.item(i)):
            count+=1
    return(count/len(prediction)*100)
    
data=pd.read_csv(r"C:\Users\pawan_300\Desktop\Project work\ml files\project flower\Demo.csv")
test,train=train_test_split(data,test_size=0.3)
y=np.matrix(test[test.columns[4]])
test=test.drop([test.columns[4]],axis=1)