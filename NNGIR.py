import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
class NNGIR():
    data=0
    target=0
    def setData(self,data,target):
        self.data=data
        self.target=target
    def NaNSearching(self,data):
        '''
        NaNSearching算法
        参数：
            暂无
        返回：
            supK:
            NN:带每个点的最supk个临近节点
            nb:

        '''

        kdt=KDTree(data,leaf_size=int(len(data)*1.5),metric="euclidean")
        r=1
        nb=np.zeros(len(data),dtype=int)
        NN=[[] for i in range(len(data))]
        RNN=[[] for i in range(len(data))]
        oldNubm=0
        while True:
            #k值不同时最邻近的值的顺序不同，待解
            for i in kdt.query(data,k=r+1,return_distance=False):

                nb[i[r]]+=1
                if i[r] not in NN[i[0]]:
                    NN[i[0]].append(i[r])
                if i[0] not in RNN[i[r]]:
                    RNN[i[r]].append(i[0])
            newNumb=0
            for i in nb:
                if i==0:
                    newNumb+=1
            if newNumb==oldNubm:
                break
            r+=1
            oldNubm=newNumb
        self.supk=r

        return r,nb,NN
    def getNaNG(self,data,target):
        r,nb,NN=self.NaNSearching(data)
        V=np.arange(len(data))
        E={}
        hoe=np.zeros(len(V))#同类点边
        hee=np.zeros(len(V))#异类点边
        for x in V:
            E[x]=NN[x]
        for x in V:
            for y in NN[x]:
                if target[x]==target[y]:
                    hoe[x]+=1
                else:
                    hee[x]+=1
        return hoe,hee,nb,NN
    def noisyFilter(self,data,target):
        hoe,hee,nb,NN=self.getNaNG(data,target)
        X=np.arange(len(data))
        newX=[]
        for x in X:
            if nb[x]!=0:
                if hoe[x]>hee[x]:
                    newX.append(x)
        newData=np.zeros((len(newX),len(self.data[0])))
        newTarget=np.zeros(len(newX))
        for index,i in enumerate(newX):
            newData[index]=data[i]
            newTarget[index]=target[i]
        return newData,newTarget
    def NNGIR(self,data,target):
        hoe,hee,nb,NN=self.getNaNG(data,target)

        BS=[]
        B=[]
        S=[]
        noise=[]
        while max(nb)!=0:
            x=np.argmax(nb)
            mark=0
            for j in NN[int(x)]:
                if nb[j]==0:
                    mark=1
            if mark==0:
                S.append(x)
            nb[x]=0
            for j in NN[int(x)]:
                nb[j]=0

        for x in range(len(data)):
            if hee[x]>0:
                if hoe[x]>=hee[x]:
                    for j in NN[x]:
                        if target[j]!=target[x]:
                            BS.append(j)
                else:
                    noise.append(x)
        BS=self.setdiff(BS,noise)
        #还差一步对BS的稀疏化处理，但是对
        #公式d(xi,NN(xi,B))不理解,B我认为初始值为空，不能进行运算
        #并且这步只是提高数据压缩效率，对分类结果并无好处
        #暂时停止次步

        #算法为:
        #for xi in BS:
        #    if d(xi,NN(xi,B))>d(xi,NE(xi,X')):
        #        B.append(xi)
        #for i in B:
        # if i not in S:
        #     S.append(i)
        for i in BS:
            if i not in S:
                S.append(i)

        finalData=np.zeros((len(S),len(data[0])))
        finalTarget=np.zeros(len(S))
        for index,i in enumerate(S):
            finalData[index]=data[i]
            finalTarget[index]=target[i]
        return finalData,finalTarget

    def setdiff(self,A,B):
        result=[]
        for i in A:
            if i not in B:
                result.append(i)
        return result

if __name__=="__main__":
    path="F:\人才才能预测论文\Human_performance.xlsx"
    data_0=pd.read_excel(path,sheet_name="dataset")
    humanData=np.array(data_0)

    data=humanData[:,:-1]
    target=humanData[:,-1]
    test=NNGIR()
    #输入训练集
    test.setData(data,target)
    #对数据进行去噪
    newData,newTarget=test.noisyFilter(data,target)
    print("去噪后数据长度%d"%len(newData))
    #对去噪后数据集进行进一步压缩
    finalData,finalTarget=test.NNGIR(newData,newTarget)
    #得到压缩后的数据集，用这个数据集构建模型
    print("最终数据长度%d"%len(finalData))


    #用sklearn中的knn验证
    knn = KNeighborsClassifier()
    knn.fit(finalData, finalTarget)
    predict = knn.predict(data)
    print(predict)
    count=0
    for i in range(len(predict)):
        if predict[i]==target[i]:
            count+=1
    print("分类准确率:%2f"%(count/len(predict)))

