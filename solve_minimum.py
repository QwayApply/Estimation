import pandas as pd

depm=input()

df = pd.read_excel('extract_ori.xlsx')
df1=df.where((df['전형']==depm)&((df['합격']=='최초합')|(df['합격']=='추합'))); df1=df1.dropna()

df2=df1['대학교'].drop_duplicates()
listfromdf=df2.values.tolist()

minarray=np.array([])
for i in listfromdf:
    df2=df1.where(df['대학교']==i);
    df2=df2.dropna()
    minarray=np.append(minarray,df2['누적백분위'].max())
print(minarray)

minimumvalue=np.min(mimarray)
print(minimumvalue)