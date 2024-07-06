from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from best_values import lr_best_test
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures




# Polynomial Regression Model:-

class best_degree:


    lr_best_train=[]
    lr_best_test=[]

    try:
        for i in range(0,20):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
            lr=LinearRegression()
            lr.fit(X_train,y_train)
            lr_train_pred=lr.predict(X_train)
            lr_test_pred=lr.predict(X_test)
            lr_best_train.append(lr.score(X_train,y_train))
            lr_best_test.append(lr.score(X_test,y_test))

    except Exception as e:
        raise Exception(f'Best RandomState Error in Polynomial Regression :\n',e)


    poly_best_degree_train = []
    poly_best_degree_test=[]

    try:

        for i in range(0,10):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(lr_best_test))
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
        raise Exception(f'Best Degree Error in Polynomial Regression :\n'+str(e))
    

class Polynomial_regression(best_degree):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(lr_best_test))

    try:

        try:

            poly=PolynomialFeatures(degree=np.argmax(best_degree.poly_best_degree_train)) # type: ignore
            X_train_poly=poly.fit_transform(X_train)
            X_test_poly=poly.fit_transform(X_test)
            lr_poly=LinearRegression()
            lr_poly.fit(X_train_poly,y_train)
            poly_train_pred=lr_poly.predict(X_train_poly)
            poly_test_pred=lr_poly.predict(X_test_poly)
            poly_train_score=lr_poly.score(X_train_poly,y_train)
            poly_test_score=lr_poly.score(X_test_poly,y_test)
            poly_cross_val_score=cross_val_score(lr_poly,X,y,cv=5).mean()
            poly_tr_mae=mean_absolute_error(y_train,poly_train_pred)
            poly_tr_mse=mean_squared_error(y_train,poly_train_pred)
            poly_tr_rmsc=np.sqrt(mean_squared_error(y_train,poly_train_pred))
            poly_te_mae=mean_absolute_error(y_test,poly_test_pred)
            poly_te_mse=mean_squared_error(y_test,poly_test_pred)
            poly_te_rsme=np.sqrt(mean_squared_error(y_test,poly_test_pred))

        except Exception as e:
            raise Exception(f'Error find in Polynomial Regression :\n'+str(e))

            

        try:

            def __init__(self,poly_best_degree_train,poly_best_degree_test,poly,X_train_poly,X_test_poly,lr_poly,poly_train_pred,poly_test_pred,poly_train_score,poly_test_score,poly_cross_val_score,
                        poly_tr_mae,poly_tr_mse,poly_tr_rmsc,poly_te_mae,poly_te_mse,poly_te_rsme,lr_best_train,lr_best_test):
                    
                try:
                
                    self.poly_best_degree_train=poly_best_degree_train
                    self.poly_best_degree_test=poly_best_degree_test
                    self.poly=poly
                    self.X_train_poly=X_train_poly
                    self.X_test_poly=X_test_poly
                    self.lr_poly=lr_poly
                    self.poly_train_pred=poly_train_pred
                    self.poly_test_pred=poly_test_pred
                    self.poly_train_score=poly_train_score
                    self.poly_test_score=poly_test_score
                    self.poly_cross_val_score=poly_cross_val_score
                    self.poly_tr_mae=poly_tr_mae
                    self.poly_tr_mse=poly_tr_mse
                    self.poly_tr_rms=poly_tr_rmsc
                    self.poly_te_mae=poly_te_mae
                    self.poly_te_mse=poly_te_mse
                    self.poly_te_rsme=poly_te_rsme
                    self.lr_best_train=lr_best_train
                    self.lr_best_test=lr_best_test

                except Exception as e:
                    raise Exception(f'Error find in Polynomial Regression at Initiat level :\n'+str(e))

            try:
                

                def poly_best_degree_test_value(self):
                    return super().poly_best_degree_test
                def poly_best_degree_train_value(self):
                    return super().poly_best_degree_train
                def poly_regression(self):
                    return self.poly
                def poly_X_train_poly(self):
                    return self.X_train_poly
                def poly_X_test_poly(self):
                    return self.X_test_poly
                def poly_lr_poly(self):
                    return self.lr_poly
                def poly_train_pred_regression(self):
                    return self.poly_train_pred
                def poly_test_pred_regression(self):
                    return self.poly_test_pred
                def poly_train_score_regression(self):
                    return self.poly_train_score
                def poly_test_score_regression(self):
                    return self.poly_test_score
                def poly_cross_val_score_regression(self):
                    return self.poly_cross_val_score
                def poly_train_mae_regression(self):
                    return self.poly_tr_mae
                def poly_train_mse_regression(self):
                    return self.poly_tr_mse
                def poly_train_rmse_regression(self):
                    return self.poly_tr_rmsc
                def poly_test_mae_regression(self):
                    return self.poly_te_mae
                def poly_test_mse_regression(self):
                    return self.poly_te_mse
                def poly_test_rmse_regression(self):
                    return self.poly_te_rsme
                def lr_best_train_poly(self):
                    return super().lr_best_train
                def lr_best_test_poly(self):
                    return super().lr_best_test
                
            except Exception as e:
                raise Exception(f'Error find in Polynomial Regression at Defining level :\n'+str(e))
            
        except Exception as e:
            raise Exception(f'Error find in Polynomial Regression at Initiat and Defining level :\n'+str(e))

    except Exception as e:
        raise Exception(f'Total Error in Polynomial Regression :\n'+str(e))

        
    