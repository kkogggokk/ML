# 03. 데이터셋 분리와 교차검증

![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-03-23%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%209.05.51.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-03-23%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%209.05.51.png)

## 3.1 데이터셋 
1. Train DataSet (훈련, 학습 데이터셋)     
 : 모델을 학습시킬 때 사용할 데이터 셋
 

2. Validation DataSet (검증 데이터셋)      
 : train dataset으로 학습한 모델의 성능을 측정하기 위한 데이터 셋 
 
 
3. Test DataSet (평가 데이터 셋) 
 : 모델의 성능을 최종적으로 측정하기 위한 데이터 셋       
 : Test 데이터셋은 마지막에 모델의 성능을 측정하는 용도로 **한번만 사용**되야 한다.


- 학습과 평가를 반복하다 보면 모델이 검증때 사용한 데이터셋에 과적합되어 새로운 데이터에 대한 성능이 떨어진다. 그래서 데이터셋을 train, validation, test 로 나눠 train와 validation로 모델을 최적화 한 뒤 마지막에 test로 최종 평가를 한다.

- 데이터셋을 나누는 방식은 holdout 방식과 K-Fold Cross Validation(K겹교차) 방식이 있다. 

## 3.2 Holdout 방식 
- 데이터셋을 train, validation, test 3개로 나눈다. 
- sklearn.model_selection.train_test_split() 함수를 사용한다.
- 3개의 데이터셋을 만들기 위해서는     
    1) train set과 test set으로 나눈다.     
    2) train set에서 validation set을 나눠 3개를 만든다.       
![image.png](attachment:image.png)

### 3.2.1 설명필요 
todo

- train_test_split()
    - test_size :
    - stratify :


```python
from sklearn.datasets import load_iris # 데이터셋 임포트
from sklearn.tree import DecisionTreeClassifier #  모델임포트
from sklearn.model_selection import train_test_split # 분리 
from sklearn.metrics import accuracy_score # 평가지표 

# 0. Dataset Load
iris = load_iris()
X, y = iris['data'], iris['target']
print('- X.shape :', X.shape)
print('- y.shape :', y.shape)

# 1. 분리
# 1.1) train_data_set, test_data_set으로 분리 
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                   test_size = 0.2,
                                                   stratify = y,
                                                   random_state = 1)
# 1.2) train_data_set, validation_data_set으로 분리 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size = 0.2,
                                                  stratify = y_train,
                                                  random_state = 1)
print('- y_train shape :', y_train.shape)
print('- y_val   shape :', y_val.shape)
print('- y_test  shape :', y_test.shape)

# 2. 모델생성 : 결정트리 
tree = DecisionTreeClassifier(max_depth = 2, random_state = 1)

# 3. 모델학습
tree.fit(X_train, y_train)

# 4. 예측 및 검증
pred_train = tree.predict(X_train)
pred_val = tree.predict(X_val)

acc_train = accuracy_score(y_train, pred_train)
acc_val = accuracy_score(y_val, pred_val)

print('- train       accuracy :', acc_train)
print('- validataion accuracy :', acc_val)

# 5. 최종 평가 
pred_test = tree.predict(X_test)
acc_test = accuracy_score(y_test, pred_test)
print('- 최종 검증 결과(test)    :', acc_test)
```

    - X.shape : (150, 4)
    - y.shape : (150,)
    - y_train shape : (96,)
    - y_val   shape : (24,)
    - y_test  shape : (30,)
    - train       accuracy : 0.96875
    - validataion accuracy : 0.9166666666666666
    - 최종 검증 결과(test)    : 0.9333333333333333


### 3.2.2 holdout 방식 단점
- train_set, test_set이 어떻게 나누느냐에 따라서 결과가 달라진다.
- 데이터의 양이 충분히 많을때는 변동성이 흡수되어 괜찮으나 수천건정도로 적을때는 문제가 발생할 수 있다. 
- 따라서 데이터셋의 양이 적을 경우 적합하지 않다. 
- 이를 해결하기 위해 나온 방법이 K겹 교차검증 방식이다. 

## 3.2 K-Fold Cross Validation
- 전체 데이터셋을 K개로 나누다. 그 중 한개를 test_set(검증셋)으로 나머지는 train_set(훈련셋)으로 하여 모델을 학습시키고 평가한다. 
- 나눠진 K개의 데이터셋을 K번 반복하여 모델 학습시킨 뒤 나온 평가지표들의 평균으로 모델의 성능을 평가한다. 
- 종류는 

### 3.2.1 K-Fold
- 지정한 갯수 K만큼 분할한다. 

> ### 참고 ) Kfold.split()
> - TODO 


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 0. Dataset Load
iris = load_iris()
X, y = iris['data'], iris['target']

kfold = KFold(n_splits = 3) #
acc_train_list = [] # train_set 정확도를 저장할 리스트 선언
acc_test_list = [] # test_set 정확도를 저장할 리스트 선언 

for train_index, test_index in kfold.split(X):
    # 1. 분리
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    # 2. 모델 생성
    tree = DecisionTreeClassifier(max_depth=2)
    
    # 3. 모델 학습
    tree.fit(X_train, y_train)
    
    # 4. 예측 및 검증
    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)
    
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)
    
# 5. 최종 평가 
print('- train 정확도 ',acc_train_list)
print('- test  정확도 ',acc_test_list)
print('- train 정확도 평균:',np.mean(acc_train_list))
print('- test  정확도 평균:',np.mean(acc_test_list))
```

    - train 정확도  [0.96, 1.0, 1.0]
    - test  정확도  [0.0, 0.0, 0.0]
    - train 정확도 평균: 0.9866666666666667
    - test  정확도 평균: 0.0


#### 3.2.1.1 K-Fold문제점 
- test정확도의 결과가 0이 나왔다. 그 이유는 ? 
- 즉, 원 데이터셋의 row순서대로 분할하기 때문에 불균형 문제가 발생할 수 있다. 
- 이 문제를 해결하기 위해 Stratified K-Fold 을 알아보도록 하자. 


```python
# 인덱스의 값을 확인하게 되면 ? --> 이게 의미하는게 뭔지 설명해줘야함. 
print('- train index 값 :\n',y[train_index])
print('- test index 값 :\n',y[test_index])
```

    - train index 값 :
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    - test index 값 :
     [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2]


### 3.2.2 Stratified K-Fold
- 나뉜 fold들에 label들이 같은(또는 거의 비슷한) 비율로 구성되도록 나눈다. 


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 0. Dataset Load
iris = load_iris()
X, y = iris['data'], iris['target']

s_fold = StratifiedKFold(n_splits = 3)

acc_train_list = []
acc_test_list = []


for train_index, test_index in s_fold.split(X, y):
    # 1. 분리
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    # 2. 모델 생성
    tree = DecisionTreeClassifier(max_depth = 2)
    
    # 3. 모델 학습
    tree.fit(X_train, y_train)
    
    # 4. 예측 및 검증
    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)
    
    acc_train_list.append(accuracy_score(y_train, pred_train))
    acc_test_list.append(accuracy_score(y_test, pred_test))

# 5. 최종 평가 
print('- train 정확도    :',acc_train_list)
print('- test  정확도    :',acc_test_list)

print('- train 정확도 평균 :',np.round(np.mean(acc_train_list),2))
print('- test  정확도 평균 :',np.round(np.mean(acc_test_list),2))
```

    - train 정확도    : [0.95, 0.98, 0.96]
    - test  정확도    : [0.96, 0.92, 0.92]
    - train 정확도 평균 : 0.96
    - test  정확도 평균 : 0.93


## 3.3 cross_val_score()
- 데이터셋을 K개로 나누고 K번 반복하여 평가하는 작업을 해주는 함수 
- cross_val_score 함수의 매개변수
    - estimator: 학습할 평가모델 지정
    - X: feature
    - y: label
    - scoring: 평가지표
    - cv: 나눌 개수 (K)
- 반환값: 각 반복 시 나오는 평가점수들을 array로 반환한다.  


```python
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris['data'], iris['target']
tree = DecisionTreeClassifier(max_depth = 2)

scores = cross_val_score(
    estimator = tree, 
    X = X, 
    y = y,
    scoring = 'accuracy',
    cv = 3)

print('- 평가     점수 :', scores)
print('- 평가 평균 점수 :', np.round(np.mean(scores), 2))
```

    - 평가     점수 : [0.96 0.92 0.92]
    - 평가 평균 점수 : 0.93


--
# References      
> - 김성환, 엔코아 플레이데이터 (2021, 인공지능 개발자 3기 과정)
