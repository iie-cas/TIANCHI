# coding: utf-8

import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import time
from tool import rf
import re
import os
import math


# 进程数(将DataFrame划分成 SPLITS 块，每块交给一个进程处理)
SPLITS = 12
# 候选集样例数(用余弦相似度选出候选集)
CANDIDATE_NUM = 5


def cos_sim(vector_a, vector_b):
    """
    计算余弦相似度
    vector_a: 向量 a
    vector_b: 向量 b
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


class TianChi:

    def __init__(self):
        """
        初始化函数，加载数据，连接数据
        """
        self.shop_info = pd.read_csv('./data/shop_info.csv', sep=',', encoding='utf8', engine='c')
        self.user_info = pd.read_csv('./data/user_shop_behavior.csv', sep=',', encoding='utf8', engine='c')
        self.evl_data = pd.read_csv('./data/evaluation.csv', sep=',', encoding='utf8', engine='c')
        self.train_data = pd.merge(self.user_info, self.shop_info, on=['shop_id'])
    
    def f_cossim_candidate(self, row):
        cossim_candidate = {}
        user_wifi_name = [x[0] for x in row['wifi_infos']]
        user_wifi_vector = [x[1] for x in row['wifi_infos']]
        for shop_id in  self.shops_wifi:
            shop_wifi_vector = []
            for wifi_name in user_wifi_name:
                if wifi_name in  self.shops_wifi[shop_id]:
                    shop_wifi_vector.append( self.shops_wifi[shop_id][wifi_name])
                else:
                    shop_wifi_vector.append(0)
            shop_wifi_vector = np.array(shop_wifi_vector)
            cossim = cos_sim(user_wifi_vector, shop_wifi_vector)
            if np.isnan(cossim):
                continue
            else:
            	cossim_candidate[shop_id] = cossim
        cossim_candidate = [x[0] for x in sorted(cossim_candidate.items(), key=lambda x: x[1], reverse=True)[0:CANDIDATE_NUM]]   
        row['cossim_candidate'] = set(cossim_candidate)
        return row

    def apply_cossim_candidate(self, df_part):
        return df_part.apply(self.f_cossim_candidate, axis=1)

    def cossim_candidate_process(self, df):
        df = df.reindex(columns=df.columns.tolist() + ['cossim_candidate'])
        df_parts = np.array_split(df, SPLITS)
        with Pool(processes=SPLITS) as pool:
            df_parts_r = pool.map(self.apply_cossim_candidate, df_parts)
        df = pd.concat(df_parts_r)
        return df
    
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
    
    @staticmethod
    def wifi_process(df, wifi_dict):
        df = df.reindex(columns=df.columns.tolist() + ['wifi_' + str(i) for i in range(len(wifi_dict))])
        for index, row in df.iterrows():
            for wifi_info in row['wifi_infos']:
                wifi_name = wifi_info[0]
                wifi_isty = wifi_info[1]
                if wifi_name in wifi_dict:
                    df.loc[index, 'wifi_'+str(wifi_dict[wifi_name])] = wifi_isty
        return df
    
    @staticmethod
    def wifi_info_process(wifi_info):
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

        # 挑选每个商铺的 WIFI(用于余弦相似度的计算)
        self.shops_wifi = {}
        for shop_id in self.shops_wifi_isty:
            shop_wifi_isty = self.shops_wifi_isty[shop_id]
            shop_wifi_tf = self.shops_wifi_tf[shop_id]
            shop_wifi = {}
            for wifi_name in shop_wifi_isty:
                if shop_wifi_tf[wifi_name] >= 0.02:
                    shop_wifi[wifi_name] = shop_wifi_isty[wifi_name]
            self.shops_wifi[shop_id] = shop_wifi_isty
        
    def chunks(self, arr, m):
        """
        将一个列表等分成 m 份
        """
        n = int(math.ceil(len(arr) / float(m)))
        return [arr[i:i + n] for i in range(0, len(arr), n)]
    
    def run_proc(self, candidate_chunks, evl_mall_df, file_name):
        """
        对每个样例进行预测调用的函数
        """
        result = {}
        step = 1
        for candidates in candidate_chunks:
            # 并没有输出太多信息，如果想看具体的信息，可以自行输出一下，这里删减了
            print('step:', step)
            step += 1
            test_df = evl_mall_df[evl_mall_df['cossim_candidate'] == candidates]
            columns = ['longitude', 'latitude', 'wifi_infos']
            row_ids = list(test_df['row_id'])
            test_df = test_df[columns]
            shop_ids = list(pd.concat([self.shops[shop_id]['shop_id'] for shop_id in candidates]))
            shop_df = pd.concat([self.shops[shop_id][columns] for shop_id in candidates])
            shop_df = pd.concat([shop_df, test_df])
            wifi_dict = {}
            wifi_num = 0
            for shop_id in candidates:
                '''
                构造候选集SHOP的WIFI特征
                这个部分有很多种方式（不同的方式，可以按照TF来构造，也可以只按照出现次数来构造等），
                时间所限，这里并没有全部尝试完成
                '''
                shop_wifi_tf = self.shops_wifi_tf[shop_id]
                # shop_wifi_count = self.shops_wifi_count[shop_id]
                # shop_wifi_count = sorted(shop_wifi_count.items(), key=lambda x: x[1], reverse=True)
                # shop_wifi_count = [x[0] for x in shop_wifi_count[:int(len(shop_wifi_count)*0.2)]]
                for wifi_name in shop_wifi_tf:
                    if wifi_name not in wifi_dict and shop_wifi_tf[wifi_name]>=0.02:
                        wifi_dict[wifi_name] = wifi_num
                        wifi_num += 1
                # for wifi_name in shop_wifi_count:
                    # if wifi_name not in wifi_dict:
                        # wifi_dict[wifi_name] = wifi_num
                        # wifi_num += 1
            shop_df = TianChi.wifi_process(shop_df, wifi_dict)
            columns = ['longitude', 'latitude'] + ['wifi_' + str(i) for i in range(len(wifi_dict))]  
            shop_df = shop_df[columns]
            shop_df = shop_df.fillna(0)
            X = np.asarray(shop_df, dtype=np.float64)
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(X)
            X = min_max_scaler.transform(X)
            X_train = X[:len(shop_ids)]
            X_test = X[len(shop_ids):]
            # 预测 shop_ids
            shop_ids_predict = rf.train(X_train, shop_ids, X_test)
            for i, row_id in enumerate(row_ids):
                result[row_id] = shop_ids_predict[i]
        
        with open('./runs/' + file_name, 'w') as f:
            for row_id in result:
                f.write('%s,%s\n' %(row_id, result[row_id]))

    
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
            train_mall_df['wifi_infos'] = train_mall_df['wifi_infos'].apply(lambda x: [TianChi.wifi_info_process(wifi.split('|')) for wifi in x.split(';')])
            evl_mall_df['wifi_infos'] = evl_mall_df['wifi_infos'].apply(lambda x: [TianChi.wifi_info_process(wifi.split('|')) for wifi in x.split(';')])
             # 提取训练集标签和测试集行号
            row_ids = list(evl_mall_df['row_id'])
            shop_ids = list(train_mall_df['shop_id'])
            # 提取需要的列
            train_columns = ['longitude', 'latitude', 'wifi_infos', 'shop_id']
            evl_columns = ['longitude', 'latitude', 'wifi_infos', 'row_id']
            train_mall_df = train_mall_df[train_columns]
            evl_mall_df = evl_mall_df[evl_columns]
            # mall 数据结构初始化
            self.mall_init(mall_id, train_mall_df, evl_mall_df)
            # 连接train_mall_df和evl_mall_df进行预处理
            df = pd.concat([train_mall_df, evl_mall_df])
            df_temp = self.get_wifi_vector(df)
            columns = ['longitude', 'latitude'] + ['wifi_' + str(i) for i in range(len(self.wifi))]  
            df_temp = df_temp[columns]
            df_temp = df_temp.fillna(0)
            X = np.asarray(df_temp, dtype=np.float64)
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(X)
            X = min_max_scaler.transform(X)
            # 分离出训练集和测试集
            X_train = X[:len(shop_ids)]
            X_test = X[len(shop_ids):]
            # 第一步预测：用随机森林进行预测，筛选出单个SHOP预测概率大于0.8的样本，并确定预测结果为最终结果
            results = {}
            results_all = {}
            shop_ids_prob, lbl =rf.train_prob(X_train, shop_ids, X_test)
            for i, row in enumerate(shop_ids_prob):
                # 统计单个店铺概率概率大于0.8结果
                for k, prob in enumerate(row):
                    if prob >= 0.8:
                        shop_id = lbl.inverse_transform(int(k))
                        row_id = row_ids[i]
                        results[row_id] = shop_id
                        break
                # 统计随机森林预测的所有结果
                k_temp = 0
                prob_temp = 0
                for k, prob in enumerate(row):
                    if prob > prob_temp:
                        k_temp = k
                        prob_temp = prob
                shop_id = lbl.inverse_transform(int(k_temp))
                row_id = row_ids[i]
                results_all[row_id] = shop_id

            # 筛选出第一步没有确定商铺的测试集样例
            for row_id in results:
                df.loc[df.row_id == row_id, 'shop_id'] = results[row_id]
            evl_mall_df = df[df.shop_id.isnull()]
            row_ids = evl_mall_df['row_id']
            
            # 计算测试集每个样例的候选集
            # 候选集的挑选仅使用了用户的WIFI列表和每个店铺的WIFI列表的余弦相似度
            evl_mall_df = self.cossim_candidate_process(evl_mall_df)
            
            # 统计所有的候选集集合
            candidates_list = []
            for candidates in evl_mall_df['cossim_candidate']:
                # 其中 candidates 的类型为 set
                if candidates not in candidates_list:
                    candidates_list.append(candidates)
            
            # 过滤长度为0和长度为1的候选集集合
            for candidates in candidates_list:
                test_df = evl_mall_df[evl_mall_df['cossim_candidate'] == candidates]
                # 长度为0，说明用户的WIFI列表里的WIFI在之前的训练集中没有出现过
                if len(candidates) == 0:
                    for index, row in test_df.iterrows():
                        row_id = row['row_id']
                        results[row_id] = results_all[row_id]
                    candidates_list.remove(candidates)
                # 长度为1，说明用户的WIFI列表仅与一家商铺的WIFI匹配
                elif len(candidates) == 1:
                    for index, row in test_df.iterrows():
                        row_id = row['row_id']
                        results[row_id] = list(candidates)[0]
                    candidates_list.remove(candidates)
            print('\n', 'TOTAL STEPS:', len(candidates_list), '\n')
            
            # 划分候选集集合
            candidate_chunks = self.chunks(candidates_list, SPLITS)
            file_list = [str(i)+'.csv' for i in range(SPLITS)]

            # 多进程处理各个候选集情况
            p = Pool(SPLITS)
            for i in range(SPLITS):
                p.apply_async(self.run_proc, args=(candidate_chunks[i], evl_mall_df, file_list[i]))
            p.close()
            # 全部进程结束才执行下面的汇总结果的代码
            p.join()

            # 汇总结果
            fw = open('./mall_results/' + str(mall_id) + 'result.csv', 'w')
            fw.write('row_id,shop_id\n')
            for row_id in results:
                fw.write('%s,%s\n' %(row_id, results[row_id]))
            for filename in file_list:
                with open('./runs/' + filename, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() != '':
                            fw.write(line)
            fw.close()
   
    
if __name__ == '__main__':
    data = TianChi()
    data.run()
