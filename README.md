# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages. 
2.Analyse the data. 
3.Use modelselection and Countvectorizer to preditct the values. 
4.Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BOJA RAJA G
RegisterNumber: 212225230036 
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
<img width="793" height="445" alt="image" src="https://github.com/user-attachments/assets/9d1221e8-0bdb-4b23-ac9c-2f04523df888" />
<img width="115" height="34" alt="image" src="https://github.com/user-attachments/assets/015362fc-9808-4450-b807-328d374bc6be" />
<img width="81" height="35" alt="image" src="https://github.com/user-attachments/assets/4387bcf8-0146-4c38-81a3-1c5623816dc3" />
<img width="86" height="34" alt="image" src="https://github.com/user-attachments/assets/9e833bcc-32f1-4d87-b490-cf60b913edc2" />
<img width="1247" height="215" alt="image" src="https://github.com/user-attachments/assets/ff71129d-e7c3-47af-b136-193d127404a7" />
<img width="97" height="31" alt="image" src="https://github.com/user-attachments/assets/5fa8e23f-b1ad-419f-a66d-92a6fd87a97a" />
<img width="678" height="46" alt="image" src="https://github.com/user-attachments/assets/64e40bae-608a-4f37-946c-d6ce90ed5edc" />
<img width="135" height="68" alt="image" src="https://github.com/user-attachments/assets/960d93f3-212c-49f3-a2d4-5d9637fe576d" />
<img width="599" height="199" alt="image" src="https://github.com/user-attachments/assets/a9b1ae13-3d13-480b-8758-037c6f8894fd" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
