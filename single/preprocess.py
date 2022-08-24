import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_year(s):
    year = int(s[0])+1
    if year==10:
        year=0
    return year

brand_lst = ['Q235B','CCS A','AB/A','SS400']
def get_brand(s):
    return brand_lst.index(s)

def get_data(data_path):
    df = pd.read_csv(data_path)
    # df = df.drop([df.columns[0],'试批号','钢种','粗轧压缩比','精轧压缩比','Ceq','Pcm','温度差'], axis=1)
    df = df.drop([df.columns[0],'试批号','钢种'], axis=1)
    df['year'] = df['材料'].apply(get_year)
    df['brand'] = df['牌号'].apply(get_brand)
    df = df.drop(['材料','牌号'], axis=1)
    df = df.drop(df[(df['出炉温度']==0) | (df['加热时间']==0)].index)
    if str(df.loc[1,'粗轧压下率'])[-1]=='%':
        df['粗轧压下率'] = df['粗轧压下率'].apply(lambda x: float(x[:-1])/100)
    df = df.reset_index(drop=True)

    return df

pca = PCA(n_components=15)
std = StandardScaler()

def gen_train(df):
    X = df.iloc[:,3:]
    std.fit(X)
    X = std.transform(X)
    # pca.fit(X)
    # X = pca.transform(X)

    X = pd.DataFrame(X)
    return pd.merge(df.loc[:,['实验屈服值', '实验抗拉值', '实验伸长率']], X, left_index=True, right_index=True)

def gen_test(df):
    X = df.iloc[:,3:]
    X = std.transform(X)
    # X = pca.transform(X)
    X = pd.DataFrame(X)
    return pd.merge(df.loc[:,['实验屈服值', '实验抗拉值', '实验伸长率']], X, left_index=True, right_index=True)


a = get_data('../data/a.csv')
b = get_data('../data/b.csv')
c = get_data('../data/c.csv')

train = pd.concat([a, b, c]).reset_index(drop=True)
test = get_data('../data/test.csv')

dataset = pd.concat([train, test])
dataset = pd.get_dummies(dataset, columns = ['year','brand'])
train = dataset.iloc[:train.shape[0],:]
test = dataset.iloc[train.shape[0]:,:]

train = gen_train(train)
print('train:')
print(train)
test = gen_test(test)
print('test:')
print(test)
train.to_csv('./data/train.csv')
test.to_csv('./data/test.csv')