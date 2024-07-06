from data import df
import seaborn as sns
import matplotlib.pyplot as plt

class News_IQR:
        
    try:

        q1_news=df['newspaper'].quantile(0.25)
        q2_news=df['newspaper'].quantile(0.50)
        q3_news=df['newspaper'].quantile(0.75)

        iqr_news=q3_news-q1_news

        upper_news=q3_news+(1.5*iqr_news)

        lower_news=q1_news-(1.5*iqr_news)

    except Exception as e:
        raise Exception(f'Error find in News_IQR from data cleaning :\,'+str(e))
        
    def __init__(self,q1_news,q2_news,q3_news,iqr_news,upper_news,lower_news):
            
        try:
        
            self.q1_news=q1_news
            self.q2_news=q2_news
            self.q3_news=q3_news
            self.iqr_news=iqr_news
            self.upper_news=upper_news
            self.lower_news=lower_news

        except Exception as e:
            raise Exception(f'Error find in News_IQR initilization from data cleaning :\n'+str(e))
        
    try:

        def q1_column_news(self):
            return self.q1_news
        def q2_column_news(self):
            return self.q2_news
        def q3_column_news(self):
            return self.q3_news
        def iqr_column_news(self):
            return self.iqr_news
        def upper_column_news(self):
            return self.upper_news
        def lower_column_news(self):
            return self.lower_news
        
    except Exception as e:
        raise Exception(f'Error find in News_IQR dening from data cleaning :\n'+str(e))
    

df[(df['newspaper']<News_IQR.upper_news) & (df['newspaper']>News_IQR.lower_news)] # type: ignore

df[(df['newspaper']>News_IQR.upper_news) | (df['newspaper']<News_IQR.lower_news)] # type: ignore
    

df['newspaper']=df['newspaper'].clip(lower=News_IQR.lower_news, upper=News_IQR.upper_news) # type: ignore


sns.boxplot(df['newspaper']) # type: ignore
plt.title('Newspaper column Box plot after outliers')
plt.savefig("C:/AI&ML Engineer/Projects/Regression/Polynomial Regression/plots/news_box_after_outlier.png")




