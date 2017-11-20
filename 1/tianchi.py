# coding: utf-8

import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import xgb
from rf import RF


# 12个进程
SPLITS = 12

class TianChi:
    def __init__(self):
        """
        初始化函数，加载数据，连接数据
        """
        self.shop_info = pd.read_csv('./data/shop_info.csv', sep=',', encoding='utf8', engine='c')
        self.user_info = pd.read_csv('./data/user_shop_behavior.csv', sep=',', encoding='utf8', engine='c')
        self.evl_data = pd.read_csv('./data/evaluation.csv', sep=',', encoding='utf8', engine='c')
        self.train_data = pd.merge(self.user_info, self.shop_info, on=['shop_id'])

    def wifi_info_process(self, wifi_info):
        """
        预处理 wifi_infos 字段
        """
        wifi_name = wifi_info[0]
        wifi_isty = int(wifi_info[1])
        wifi_conn = wifi_info[2]
        if wifi_isty <= -100:
            wifi_isty = 0
        else:
            wifi_isty = (wifi_isty + 100) / 100.0
        if wifi_conn == 'true':
            wifi_conn = True
        else:
            wifi_conn = False
        return [wifi_name, wifi_isty, wifi_conn]
    
    def mall_init(self, mall_id, train_mall_df, evl_mall_df):
        """
        商场的数据结构初始化
        mall_id: 商场ID
        train_mall_df: 训练集 DataFrame
        evl_mall_df: 测试集 DataFrame
        """
        self.shops = {}
        shop_list = self.shop_info[self.shop_info.mall_id == mall_id].shop_id.unique()
        for shop_id in shop_list:
            self.shops[shop_id] = train_mall_df[train_mall_df.shop_id == shop_id]
        print('MALL ID: %s\nTRAIN NUM: %s\nEVL_NUM: %s\nSHOP_NUM: %s' %(mall_id, train_mall_df.shape[0], evl_mall_df.shape[0], len(shop_list)))
        # 统计每个SHOP的WIFI数和每个SHOP的每个WIFI的强度和
        self.shops_wifi_count = {}
        self.shops_wifi_isty = {}
        for shop_id in self.shops:
            shop = self.shops[shop_id]
            shop_wifi_count = {}
            shop_wifi_isty = {}
            for index, row in shop.iterrows():
                for wifi_info in row['wifi_infos']:
                    wifi_name = wifi_info[0]
                    wifi_isty = wifi_info[1]
                    if wifi_name not in shop_wifi_count:
                        shop_wifi_count[wifi_name] = 1
                        shop_wifi_isty[wifi_name] = wifi_isty
                    else:
                        shop_wifi_count[wifi_name] += 1
                        shop_wifi_isty[wifi_name] += wifi_isty
            self.shops_wifi_count[shop_id] = shop_wifi_count
            self.shops_wifi_isty[shop_id] = shop_wifi_isty

        # 求每家商铺的 WIFI 的平均强度
        for shop_id in self.shops_wifi_isty:
            shop_wifi_isty = self.shops_wifi_isty[shop_id]
            shop_wifi_count = self.shops_wifi_count[shop_id]
            for wifi_name in shop_wifi_isty:
                shop_wifi_isty[wifi_name] = float(shop_wifi_isty[wifi_name]) / (shop_wifi_count[wifi_name])
            self.shops_wifi_isty[shop_id] = shop_wifi_isty

        # 商场中每家商铺的每个WIFI的TF值
        self.shops_wifi_tf = {}
        for shop_id in self.shops_wifi_count:
            shop_wifi_count = self.shops_wifi_count[shop_id]
            shop_wifi_tf = {}
            total_num = float(sum(shop_wifi_count.values()))
            for wifi_name in shop_wifi_count:
                shop_wifi_tf[wifi_name] = shop_wifi_count[wifi_name] / total_num
            self.shops_wifi_tf[shop_id] = shop_wifi_tf
        
        # 统计整个商场中，每个WIFI的出现次数
        train_wifi_count = {}
        for wifi_infos in train_mall_df['wifi_infos']:
            for wifi_info in wifi_infos:
                wifi_name = wifi_info[0]
                if wifi_name not in train_wifi_count:
                    train_wifi_count[wifi_name] = 1
                else:
                    train_wifi_count[wifi_name] += 1

        # 统计商场中WIFI出现次数大于10的WIFI
        train_wifi_gt10 = set()
        for wifi_name in train_wifi_count:
            if train_wifi_count[wifi_name] >= 10:
                train_wifi_gt10.add(wifi_name)

        # 筛选出做特征的WIFI
        self.wifi = {}
        wifi_num = 0
        # 1. 筛选出每个SHOP的WIFI的TF值大于0.02的WIFI
        for shop_id in self.shops_wifi_tf:
            shop_wifi_tf = self.shops_wifi_tf[shop_id]
            for wifi_name in shop_wifi_tf:
                if shop_wifi_tf[wifi_name] >= 0.02:
                    if wifi_name not in self.wifi:
                        self.wifi[wifi_name] = wifi_num
                        wifi_num += 1
        # 2. 筛选出整个商场中WIFI出现次数大于10的WIFI
        for wifi_name in train_wifi_gt10:
            if wifi_name not in self.wifi:
                self.wifi[wifi_name] = wifi_num
                wifi_num += 1
        # 3. 筛选出整个商场中WIFI出现次数的TOP10%
        wifi_perc10 = sorted(train_wifi_count.items(), key=lambda d: d[1], reverse=True)
        wifi_perc10 = [x[0] for x in wifi_perc10[:int(len(train_wifi_count)*0.10)]]
        for wifi_name in wifi_perc10:
            if wifi_name not in self.wifi:
                self.wifi[wifi_name] = wifi_num
                wifi_num += 1
        self.wifi_num = wifi_num
        print('WIFI NUM:', self.wifi_num) 
    
    def f_wifi(self, row):
        for wifi_info in row['wifi_infos']:
            wifi_name = wifi_info[0]
            wifi_intensity = wifi_info[1]
            if wifi_name in self.wifi:
                row[-1 - self.wifi[wifi_name]] = wifi_intensity
        return row

    def apply_f_wifi(self, df):
        return df.apply(self.f_wifi, axis=1, raw=True)

    def get_wifi_vector(self, df):
        df_temp = pd.DataFrame(columns=['wifi_' + str(i) for i in range(self.wifi_num)])
        df = pd.concat([df, df_temp], axis=1)
        df_parts_temp = np.array_split(df, SPLITS)
        with Pool(processes=SPLITS) as pool:
            df_parts = pool.map(self.apply_f_wifi, df_parts_temp)
        df = pd.concat(df_parts)
        return df
    
    def run(self):
        
        mall_list = self.shop_info.mall_id.unique()
        for mall_id in mall_list:
            if mall_id != 'm_6803':
                continue
            # 提取训练集数据和验证集数据
            train_mall_df = self.train_data[self.train_data.mall_id == mall_id]
            evl_mall_df = self.evl_data[self.evl_data.mall_id == mall_id]
            train_mall_df.rename(columns={'longitude_x': 'longitude', 'latitude_x': 'latitude'}, inplace=True)
            # wif_infos 字段的预处理
            train_mall_df['wifi_infos'] = train_mall_df['wifi_infos'].apply(lambda x: [self.wifi_info_process(wifi.split('|')) for wifi in x.split(';')])
            evl_mall_df['wifi_infos'] = evl_mall_df['wifi_infos'].apply(lambda x: [self.wifi_info_process(wifi.split('|')) for wifi in x.split(';')])
            # 提取训练集标签和测试集行号
            row_ids = list(evl_mall_df['row_id'])
            shop_ids = list(train_mall_df['shop_id'])
            # 提取需要的列
            train_columns = ['longitude', 'latitude', 'wifi_infos', 'shop_id']
            evl_columns = ['longitude', 'latitude', 'wifi_infos', 'row_id']
            train_mall_df = train_mall_df[train_columns]
            evl_mall_df = evl_mall_df[evl_columns]
            # mall 数据结构初始化
            self.mall_init(mall_id, train_mall_df,evl_mall_df)
            # 连接train_mall_df和evl_mall_df进行预处理
            df = pd.concat([train_mall_df, evl_mall_df])
            df = self.get_wifi_vector(df)
            columns = ['longitude', 'latitude'] + ['wifi_' + str(i) for i in range(len(self.wifi))]  
            df = df[columns]
            df = df.fillna(0)
            X = np.asarray(df, dtype=np.float64)
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(X)
            X = min_max_scaler.transform(X)
            # 分离出训练集和测试集
            X_train = X[:len(shop_ids)]
            X_test = X[len(shop_ids):]
            rf = RF()
            rf.train(mall_id, X_train, shop_ids, X_test, row_ids)
            xgb.analyse(mall_id, X_train, shop_ids)
            xgb.train(mall_id, X_train, shop_ids, X_test, row_ids)
            print('='*120)


if __name__ == '__main__':
    data = TianChi()
    data.run()
