from data_cleaning import df
import numpy as np
import pandas as pd





df['tv_power']=df['TV']**(1.09)
print('power',df['tv_power'].skew())

df['radio_root']=(df['radio'])**(1/1.11)
print('Radio',df['radio_root'].skew())

df['news_root']=(df['newspaper'])**(1/2.027)
print('Newspaper',df['news_root'].skew())


