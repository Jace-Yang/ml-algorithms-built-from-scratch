# 1 Input


```python
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.datasets import fetch_lfw_people 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
  
import numpy as np 
```


```python
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.5) 
```

**Files display：**


```python
from IPython.display import Image
Image(filename='./image/files.png',width=400)
```




![png](./image/output_4_0.png)



- Everyone has a few photos, according to `min_faces_per_Person` I select the minimum number of faces per person.


```python
lfw_people.data.shape
```




    (3023, 2914)




```python
lfw_people.images.shape
```




    (3023, 62, 47)



# 2 Helper


```python
# introspect the images arrays to find the shapes (for plotting) 
n_samples, h, w = lfw_people.images.shape 
X = lfw_people.data 
n_features = X.shape[1] 
y = lfw_people.target 
target_names = lfw_people.target_names 
n_classes = target_names.shape[0] 
```


```python
print("Data lines:% d" % n_samples) 
print("Features:% d" % n_features)   # 与照片清晰度设置相关
print("Prople's face:% d" % n_classes) 
```

    Data lines: 3023
    Features: 2914
    Prople's face: 62



```python
def plot_faces(images, titles, h, w, n_row = 4, n_col = 10): 
    '''
    打印images中的前n_row * n_col张人脸
    '''
    plt.figure(figsize =(1.8 * n_col, 2.4 * n_row)) 
    plt.subplots_adjust(bottom = 0, left =.01, right =.99, top =.90, hspace =.35) 
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1) 
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray) 
        plt.title(titles[i], size = 12) 
        plt.xticks(()) 
        plt.yticks(()) 

# Generate true labels above the images 
def true_title(Y, target_names, i): 
    true_name = target_names[Y[i]].rsplit(' ', 1)[-1] 
    return 'True Name: % s' % (true_name) 
```

# 3 Origin face


```python
true_titles = [true_title(y, target_names, i) for i in range(y.shape[0])] 
plot_faces(X, true_titles, h, w) 
```


![png](./image/output_13_0.png)


# 4 Feature face


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42) 
print("Train size:% d｜Test size:% d" %(y_train.shape[0], y_test.shape[0])) 
```

    Train size: 2267｜Test size: 756



```python
n_components = 150
pca = PCA(n_components = n_components, svd_solver ='randomized', whiten = True).fit(X_train) 
eigenfaces = pca.components_.reshape((n_components, h, w)) 
X_train_pca = pca.transform(X_train) 
X_test_pca = pca.transform(X_test) 
```


```python
# plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_faces(eigenfaces, eigenface_titles, h, w)

plt.show()
```


![png](./image/output_17_0.png)


# 5 SVM using rbf kernel

## 5.1 parameter searching


```python
param_grid = {'C': [5e3, 1e4, 5e4],
              'gamma': [0.0005, 0.001, 0.005, 0.01]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
```


```python
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)
```

    Best estimator found by grid search:
    SVC(C=5000.0, break_ties=False, cache_size=200, class_weight='balanced',
        coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.001,
        kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)


## 5.2 Prediction


```python
# 测试集表现
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
```

                               precision    recall  f1-score   support
    
             Alejandro Toledo       0.78      0.58      0.67        12
                 Alvaro Uribe       0.62      0.56      0.59         9
              Amelie Mauresmo       0.80      0.40      0.53        10
                 Andre Agassi       0.30      0.30      0.30        10
               Angelina Jolie       0.33      0.17      0.22         6
                 Ariel Sharon       0.60      0.83      0.70        18
        Arnold Schwarzenegger       0.50      0.33      0.40        18
         Atal Bihari Vajpayee       0.75      0.43      0.55         7
                 Bill Clinton       0.50      0.33      0.40         6
                 Carlos Menem       0.75      0.60      0.67         5
                 Colin Powell       0.75      0.80      0.78        71
                David Beckham       0.22      0.50      0.31         4
              Donald Rumsfeld       0.69      0.50      0.58        36
             George Robertson       0.83      1.00      0.91         5
                George W Bush       0.62      0.84      0.71       123
            Gerhard Schroeder       0.34      0.76      0.47        17
      Gloria Macapagal Arroyo       0.92      0.75      0.83        16
                   Gray Davis       0.50      0.33      0.40         6
              Guillermo Coria       0.50      0.17      0.25        12
                 Hamid Karzai       0.50      0.40      0.44         5
                    Hans Blix       0.43      0.50      0.46         6
                  Hugo Chavez       0.85      0.74      0.79        23
                  Igor Ivanov       0.00      0.00      0.00         5
                   Jack Straw       1.00      0.33      0.50         3
               Jacques Chirac       0.58      0.44      0.50        16
                Jean Chretien       0.29      0.44      0.35         9
             Jennifer Aniston       1.00      0.20      0.33         5
            Jennifer Capriati       0.47      0.67      0.55        12
               Jennifer Lopez       0.50      0.50      0.50         4
            Jeremy Greenstock       0.67      0.36      0.47        11
                  Jiang Zemin       1.00      1.00      1.00         2
                John Ashcroft       0.78      0.58      0.67        12
              John Negroponte       0.00      0.00      0.00         8
             Jose Maria Aznar       0.50      1.00      0.67         2
          Juan Carlos Ferrero       0.60      0.75      0.67         8
            Junichiro Koizumi       0.91      0.77      0.83        13
                   Kofi Annan       1.00      0.75      0.86         8
                   Laura Bush       0.82      0.90      0.86        10
            Lindsay Davenport       0.67      0.33      0.44         6
               Lleyton Hewitt       0.47      0.70      0.56        10
    Luiz Inacio Lula da Silva       0.64      0.88      0.74         8
                Mahmoud Abbas       0.60      0.60      0.60         5
        Megawati Sukarnoputri       1.00      0.70      0.82        10
            Michael Bloomberg       0.25      0.25      0.25         4
                  Naomi Watts       1.00      0.80      0.89         5
              Nestor Kirchner       0.75      0.50      0.60        12
                  Paul Bremer       1.00      0.33      0.50         6
                 Pete Sampras       1.00      0.67      0.80         6
         Recep Tayyip Erdogan       1.00      0.50      0.67         6
                Ricardo Lagos       0.50      0.75      0.60         4
                 Roh Moo-hyun       0.88      0.78      0.82         9
             Rudolph Giuliani       0.20      0.20      0.20         5
               Saddam Hussein       1.00      1.00      1.00         6
              Serena Williams       0.77      0.71      0.74        14
            Silvio Berlusconi       0.20      0.29      0.24         7
                  Tiger Woods       0.75      0.60      0.67         5
                  Tom Daschle       1.00      0.20      0.33         5
                    Tom Ridge       0.60      0.30      0.40        10
                   Tony Blair       0.47      0.61      0.53        28
                  Vicente Fox       0.75      0.23      0.35        13
               Vladimir Putin       0.12      0.15      0.13        13
                 Winona Ryder       0.50      0.33      0.40         6
    
                     accuracy                           0.61       756
                    macro avg       0.63      0.53      0.55       756
                 weighted avg       0.64      0.61      0.60       756
    



```python
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
```

    [[7 0 0 ... 0 0 0]
     [0 5 0 ... 0 0 0]
     [0 0 4 ... 0 0 1]
     ...
     [0 0 0 ... 3 0 0]
     [0 0 0 ... 0 2 0]
     [0 0 0 ... 0 0 2]]


## 5.3 Tested result


```python
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_faces(X_test, prediction_titles, h, w)
```


![png](./image/output_26_0.png)

