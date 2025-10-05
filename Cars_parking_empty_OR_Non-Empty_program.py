import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pickle

directory_link = r"D:\Cars data\clf-data"
category = ["empty","not_empty"]
images = []
labels = []


for index,i in enumerate(category):
    for j in os.listdir(os.path.join(directory_link,i)):
        image_path = os.path.join(directory_link,i,j)
        image = cv2.imread(image_path)
        image = cv2.resize (image,(15,15))
        img = image.flatten()
        images.append(img)

        labels.append(index)

# Model Creation
x_train, x_test, y_train, y_test = train_test_split(images,labels,test_size=0.6,random_state=101)
scaler = StandardScaler()
knn  = KNeighborsClassifier()
pipe = Pipeline([('scaler',scaler),('knn',knn)])
parameters = {'knn__n_neighbors':[3, 5, 7, 9, 11, 15]}


model = GridSearchCV(pipe, parameters,cv = 4, scoring = 'accuracy' )
model.fit(images,labels)

model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

path = r"D:\Cars data\00000000_00000127.jpg"
image = cv2.imread(path)
image = cv2.resize(image,(15,15))
flatten_img = image.flatten().reshape(1, -1)
prediction = model.predict(flatten_img)

if prediction == 1:
    print("Not-Empty")
else:
    print("Empty")

