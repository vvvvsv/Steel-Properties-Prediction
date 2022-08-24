import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import os
import sys

def get_year(s):
    year = int(s[0])+1
    if year==10:
        year=0
    return year

def get_brand(s):
    return ['Q235B','CCS A','AB/A','SS400'].index(s)

def get_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop([df.columns[0],'试批号','钢种', 'Pcm'], axis=1)
    df['Pcm'] = df['C'] + df['Si']/30.0 + (df['Mn']+df['Cu']+df['Cr'])/20.0 + df['Ni']/60.0 + df['Mo']/15.0 + df['V']/10.0
    df['year'] = df['材料'].apply(get_year)
    df['brand'] = df['牌号'].apply(get_brand)
    df = df.drop(['材料','牌号'], axis=1)
    df = df.drop(df[(df['出炉温度']==0) | (df['加热时间']==0)].index)
    if str(df.loc[1,'粗轧压下率'])[-1]=='%':
        df['粗轧压下率'] = df['粗轧压下率'].apply(lambda x: float(x[:-1])/100)
    df = df.reset_index(drop=True)
    return df

def clean_data(ori_data, mode):
    data = ori_data.copy()
    cols = None
    if mode == 'del':
        cols = ['实验屈服值', '实验抗拉值', '实验伸长率', '出炉温度', '加热时间', '加热时间', '粗轧压下率', '精轧开轧温度', '终轧温度',
            'C', 'Si', 'Mn', 'AlT', 'Nb', 'V', 'Ti', 'Ni', 'Cu', 'Cr', 'Mo', 'P', 'S', 'Ceq', 'Pcm', '温度差']
    else:
        cols = ['实验屈服值', '实验抗拉值', '实验伸长率', '出炉温度', '加热时间', '加热时间', '粗轧压下率', '精轧开轧温度', '终轧温度',
            'C', 'Si', 'Mn', 'AlT', 'Nb', 'V', 'Ti', 'Ni', 'Cu', 'Cr', 'Mo', 'P', 'S']
    up_outliers = []
    down_outliers = []
    abnormal_samples = set()

    for col in cols:
        tmp = data[col].copy().to_numpy()
        tmp = tmp[np.logical_not(np.isnan(tmp))]
        u = tmp.mean()
        std = tmp.std()

        tmp = data[col].copy().to_numpy()

        error = tmp > u+3*std
        up_outliers.append(error)
        abnormal_samples = abnormal_samples.union(set(np.nonzero(error)[0]))
        tmp1 = np.where(error, np.nan, tmp)

        error = tmp < u-3*std
        down_outliers.append(error)
        abnormal_samples = abnormal_samples.union(set(np.nonzero(error)[0]))
        tmp1 = np.where(error, np.nan, tmp1)

        data[col] = tmp1

    data_c = data.dropna(axis=0, how='any')

    if mode == 'del':
        # 删除异常样本
        data = data_c
    elif mode == 'mid':
        # 用中位数填充异常样本
        for col in cols:
            mid = data_c[col].median()
            data[col].fillna(mid, inplace=True)
    elif mode == 'minmax':
        # 用最大最小值填充异常样本
        for col, up, down in zip(cols, up_outliers, down_outliers):
            mn = data_c[col].min()
            mx = data_c[col].max()
            data[col] = np.where(up, mx, data[col])
            data[col] = np.where(down, mn, data[col])
    elif mode == 'kmeans':
        # 做kmeans来填充异常样本
        std = StandardScaler()
        data_c = std.fit_transform(data_c)
        clf = KMeans(n_clusters=100).fit(data_c)

        data_new = np.zeros_like(data.to_numpy())

        for i in range(data_new.shape[0]):
            x = data.iloc[i, :].to_numpy()
            x = std.transform(x.reshape((1,-1))).reshape(-1)

            if i in abnormal_samples:
                is_nan = np.isnan(x)
                not_nan = np.logical_not(is_nan)
                point = x[not_nan]
                min_dist = None
                best_center = None
                for j in range(100):
                    center = clf.cluster_centers_[j, :]
                    center = center[not_nan]
                    dist = np.linalg.norm(point - center)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        best_center = j
                best_center = clf.cluster_centers_[best_center, :]
                x[is_nan] = best_center[is_nan]
                data_new[i, :] = x
            else:
                data_new[i, :] = x

        data_new = std.inverse_transform(data_new)
        data = pd.DataFrame(data=data_new, columns=data.columns)
        data['year'] = data['year'].astype('int')
        data['brand'] = data['brand'].astype('int')
    else:
        raise ValueError(f'clean_data: mode {mode} not found.')

    # 重新计算ceq、pcm、温度差
    data['Ceq'] = data['C'] + data['Mn']/6.0 + (data['Cr']+data['Mo']+data['V'])/5.0 + (data['Ni']+data['Cu'])/15.0
    data['Pcm'] = data['C'] + data['Si']/30.0 + (data['Mn']+data['Cu']+data['Cr'])/20.0 + data['Ni']/60.0 + data['Mo']/15.0 + data['V']/10.0
    data['温度差'] = data['精轧开轧温度'] - data['终轧温度']

    # 删除一些异常值
    data = data[data['温度差'] > 0]
    data = data[np.abs(data['板坯厚度'] / data['中间坯厚度'] - data['粗轧压缩比']) < 0.01]
    data = data[np.abs(data['中间坯厚度'] / data['成品厚度'] - data['精轧压缩比']) < 0.01]
    return data

def process_feature(ori_train, ori_test, mode='normal', max_features=20):
    # 对离散特征做onehot编码
    dataset = pd.concat([ori_train.copy(), ori_test.copy()])
    dataset = pd.get_dummies(dataset, columns = ['year','brand'])
    # 切分数据集
    train = dataset.iloc[:ori_train.shape[0], :]
    train_X = train.iloc[:,3:].to_numpy()
    train_Y = train.iloc[:,:3].to_numpy()
    test = dataset.iloc[ori_train.shape[0]:, :]
    test_X = test.iloc[:,3:].to_numpy()
    test_Y = test.iloc[:,:3].to_numpy()

    # 标准化
    std = StandardScaler()
    train_X = std.fit_transform(train_X)
    test_X = std.transform(test_X)

    # 降维
    if mode == 'normal':
        # 不降维
        pass
    elif mode =='randomforest':
        # 使用随机森林降维
        model = RandomForestRegressor()
        model.fit(train_X, train_Y[:,0])
        feature = SelectFromModel(model, threshold=-np.inf, max_features=max_features)
        train_X = feature.fit_transform(train_X, train_Y[:,0])
        test_X = feature.transform(test_X)

    elif mode == 'pca':
        # 使用PCA降维
        pca = PCA(n_components=max_features)
        train_X = pca.fit_transform(train_X)
        test_X = pca.transform(test_X)
    else:
        raise ValueError(f'process_feature: mode {mode} not found.')

    # 组合并返回
    train = pd.DataFrame(data=np.hstack((train_Y, train_X)))
    test = pd.DataFrame(data=np.hstack((test_Y, test_X)))
    return train, test

if __name__ == '__main__':
    a = get_data('../data/a.csv')
    b = get_data('../data/b.csv')
    c = get_data('../data/c.csv')

    ori_train = pd.concat([a, b, c]).reset_index(drop=True)
    ori_test = get_data('../data/test.csv')


    # for mode1 in ['del', 'mid', 'minmax', 'kmeans']:
    for mode1 in ['minmax']:
        print('clean_data:', mode1)
        train1 = clean_data(ori_train, mode=mode1)
        # for mode2, left, right in [('normal', 33, 34), ('randomforest', 27, 31), ('pca', 27, 31)]:
        for mode2, left, right in [('normal', 33, 34), ('randomforest', 25, 26), ('pca', 25, 26)]:
            for max_features in range(left, right):
                print('process_feature:',mode2, max_features)
                train, test = process_feature(train1, ori_test, mode=mode2, max_features=max_features)

                dir_path = f'./processed_data/{mode1}_{mode2}_{max_features}/'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                train_path = os.path.join(dir_path, 'train.csv')
                test_path = os.path.join(dir_path, 'test.csv')
                print(f'Saving {train_path}...')
                print(f'Saving {test_path}...')
                train.to_csv(train_path)
                test.to_csv(test_path)

