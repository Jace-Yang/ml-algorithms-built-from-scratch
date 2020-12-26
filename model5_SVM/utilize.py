import csv
from collections import Counter
from sklearn.model_selection import train_test_split

def loaddata():
    f = open('spam_or_not_spam.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    s = ['', '']
    for l in lines:
        l = l.replace('\n', '').split(',')
        if l[1] == '0':
            s[0] += l[0]
        else:
            s[1] += l[0]
    words1 = [s[0].split(' ')]
    words2 = [s[1].split(' ')]
    words = [s[0].split(' '), s[1].split(' ')]

    stopwords = ['i', 'NUMBER', 'URL', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
             'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
             'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
             'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
             'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
             've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
             'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    countlist1 = []
    countlist2 = []
    f = open('table3.txt', 'w', encoding='utf-8')
    wf = open('table_f.txt', 'w', encoding='utf-8')
    for i in range(len(words1)):
        count = Counter(words1[i])
        k = count.most_common(300)
        for i in k:
            # print(str(i[0]))
            if str(i[0]) not in stopwords:
                #print(count[str(i[0])])
                f.write(str(i[0]) + "\n")
                wf.write(str(count[(str(i[0]))]) + "\n")

    f.close()
    wf.close()

    words = []
    words2 = []
    f0 = open('table3.txt', 'r', encoding='utf-8')
    f1 = open('table_f.txt', 'r', encoding='utf-8')
    words.append(f0.read().split('\n'))
    words2.append(f1.read().split('\n'))
    f0.close()
    f1.close()
    csv.field_size_limit(500 * 1024 * 1024)
    dic = [dict(zip(words[0], range(len(words[0]))))]
    dic2 = [dict(zip(words2[0], range(len(words2[0]))))]

    x = csv.reader(open('spam_or_not_spam.csv', 'rt', encoding='utf-8'))  # filename是 name.
    X=[]
    Y=[]
    spam=0
    ham=0
    for n in x:
        input_words = n[0]
        ans = [[]]
        a = 0
        for i in dic[0]:
            if i in input_words:
                # print(words2[0][a])
                ans[0].append(int(words2[0][a]))
                a = a + 1
                # print(words2[0][22])

            else:
                ans[0].append(0)
                a = a + 1

        if n[1] == '0':
            spam+=1
            X.append(ans[0])
            Y.append(-1)
        else:
            X.append(ans[0])
            Y.append(1)
            ham+=1
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
    sum=0
    for i in y_train:
        sum+=1
    print(sum)
    return x_train,y_train, x_test, y_test

from sklearn import svm

if __name__ == '__main__':
    trainX, trainY, testX, testY = loaddata()
    print(trainY)
    # trainX,trainY=datasets.make_moons(noise=0.15, random_state=666)

    testn = len(testX)
    corr = 0
    trainer = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    trainer.fit(trainX, trainY)
    predict = trainer.predict(testX)
    print(predict)
    for j in range(testn):
        if predict[j] == testY[j]:
            corr += 1
    print(corr * 100 / testn)