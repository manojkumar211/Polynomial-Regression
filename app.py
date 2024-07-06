from joblib import load
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures

# x_new=float(input('Enter the spend value on TV :'))

# y_new=float(input('Enter the spend value on Radio :'))

df_new=pd.DataFrame({'tv_power':[45689.02313546,89765.56967864],'radio_root':[89657.0098456,99978.442699]})

# df_new=pd.DataFrame({'tv_power':[x_new],'radio_root':[y_new]})

poly=PolynomialFeatures(degree=3)
data=poly.fit_transform(df_new)

model_file='file_pickle.pickle'

final_model=pickle.load(open(model_file,'rb'))
print(final_model.predict(data))



