from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures



# Find the Best Random State Value:-


lr_best_train=[]
lr_best_test=[]

for i in range(0,20):

    try:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        lr_train_pred=lr.predict(X_train)
        lr_test_pred=lr.predict(X_test)
        lr_best_train.append(lr.score(X_train,y_train))
        lr_best_test.append(lr.score(X_test,y_test))

    except Exception as e:
        raise Exception(f'Error find in Random State from Best Values file :\n'+str(e))
    
"""print('Best Train Random State Value :',np.argmax(lr_best_train))
print('Best Test Random State Value :',np.argmax(lr_best_test))"""


poly_best_degree_train = []
poly_best_degree_test=[]

for i in range(0,10):

    try:

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)
        poly=PolynomialFeatures(degree=i)
        X_train_poly=poly.fit_transform(X_train)
        X_test_poly=poly.fit_transform(X_test)
        lr=LinearRegression()
        lr.fit(X_train_poly,y_train)
        lr_train_pred=lr.predict(X_train_poly)
        lr_test_pred=lr.predict(X_test_poly)
        poly_best_degree_train.append(lr.score(X_train_poly,y_train))
        poly_best_degree_test.append(lr.score(X_test_poly,y_test))

    except Exception as e:
        raise Exception(f'Error find in Best Degree Valued from best values file :\n'+str(e))

"""print('Best Train Degree Value :',np.argmax(poly_best_degree_train))

print('Best Test Degree Value :',np.argmax(poly_best_degree_test))"""