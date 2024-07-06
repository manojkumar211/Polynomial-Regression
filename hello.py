
# Column newspaper


"""class newspaper_column:

    newspaper_dtype=df['newspaper'].dtype
    newspaper_des=df['newspaper'].describe()
    newspaper_null=df['newspaper'].isnull().sum()
    newspaper_std=df['newspaper'].std(ddof=0)


    def __init__(self,newspaper_dtype,newspaper_des,newspaper_null,newspaper_std):

        self.newspaper_dtype=newspaper_dtype
        self.newspaper_des=newspaper_des
        self.newspaper_null=newspaper_null
        self.newspaper_std=newspaper_std


    def newspaper_column_type(self):
        return self.newspaper_dtype
    
    def newspaper_column_des(self):
        return self.newspaper_des
    
    def newspaper_column_nullvalue(self):
        return self.newspaper_null
    
    def newspaper_column_std(self):
        return self.newspaper_std
    


# Histo plot for newspaper column
    
fig,ax=plt.subplots(figsize=(10,5))
sns.histplot(data=df['newspaper'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/newspaper_histo.png")


# distribution plot for newspaper column

plt.plot(data=df['newspaper']) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/newspaper_dist.png")


# Boxplot plot for newspaper column

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['newspaper'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/newspaper_box.png")
"""