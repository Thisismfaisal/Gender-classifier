# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:51:34 2019

@author: Faisal
"""

from sklearn import tree


classifier = tree.DecisionTreeClassifier()
classifier1 = tree.ExtraTreeClassifier()


x = [ [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#train the data 
classifier.fit (x,y)
classifier1.fit(x,y)

predict = classifier.predict([[190,70,43]])
predict1 = classifier1.predict([[190,70,43]])


print(predict)
print (predict1)
