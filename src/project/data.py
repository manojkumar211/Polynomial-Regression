import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("C:/DataScience/dataset/Advertising.csv")


class data_des:
        
    try:
    
        colum=df.columns
        describ=df.describe()
        duplicate=df.duplicated().sum()
        datatype=df.dtypes
        firstrecod=df.head()
        lastrecord=df.tail()
        nullvalues=df.isnull().sum()
        datashape=df.shape
        covarince=df.cov()
        correlation=df.corr()

    except Exception as e:
        raise Exception(f'Error find in data_des from data file :\n'+str(e))

    try:

        def __init__(self,colum,describ,duplicate,datatype,firstrecod,lastrecord,nullvalues,datashape,covarince,correlation):
                
            try:

                self.colum=colum
                self.describ=describ
                self.duplicate=duplicate
                self.datatype=datatype
                self.firstrecod=firstrecod
                self.lastrecord=lastrecord
                self.nullvalues=nullvalues
                self.datashape=datashape
                self.covarince=covarince
                self.correlation=correlation

            except Exception as e:
                raise Exception(f'Error find in data_des initializing from data file :\n'+str(e))

        try:

            def data_colum(self):
                return self.colum
            def data_describ(self):
                return self.describ
            def data_duplicate(self):
                return self.duplicate
            def data_datatype(self):
                return self.datatype
            def data_firstrecod(self):
                return self.firstrecod
            def data_lastrecord(self):
                return self.lastrecord
            def data_nullvalues(self):
                return self.nullvalues
            def data_datashape(self):
                return self.datashape
            def data_covarince(self):
                return self.covarince
            def data_correlation(self):
                return self.correlation
            
        except Exception as e:
            raise Exception(f'Error find in data_des defining from data file :\n' + str(e))
        
    except Exception as e:
        raise Exception(f'Error find in class data_des from data file :\n' + str(e))
    
    
    




