from  model import MySVM
from utilize import loaddata
import matplotlib.pyplot as plt
from sklearn import datasets
trainX,trainY,testX,testY=loaddata()
#trainX,trainY=datasets.make_moons(noise=0.15, random_state=666)
testn=len(testX)
corr=0
trainer=MySVM(0.6,0.001,('linear',1),4)
trainer.fit(trainX, trainY)
predict = trainer.predict(testX)
for j in range(testn):
    if predict[j]==testY[j]:
        corr+=1
print('b:', trainer.b, '\nw:', trainer.w)
print("accuracy:"+str(corr*100/testn))
print(corr*100/testn)

testn=len(testX)

'''
C_prec=[]
C=[]
for i in range(1,10):
    corr=0
    trainer=MySVM(i*20,0.0001,('rbf',1),4)
    trainer.fit(trainX, trainY)
    predict = trainer.predict(testX)
    for j in range(testn):
        if predict[j]==testY[j]:
            corr+=1
    C_prec.append(corr*100/testn)
    C.append(i*10)

plt.plot(C,C_prec)
plt.xlabel("C")
plt.ylabel("precision")
plt.show()
'''
