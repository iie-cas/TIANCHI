# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class RF:
    def __init__(self):
        # 采用默认参数200，不寻参
        self.n_estimators = 200

    def train_prob(self, X, shop_ids, TEST):
        """
        返回预测概率
        X: 训练集 vector
        shop_ids: 训练集标签
        TEST: 测试集 vector
        """
        lbl = preprocessing.LabelEncoder()
        lbl.fit(shop_ids)
        y = lbl.transform(shop_ids)
        rf = RandomForestClassifier(n_jobs=-1, n_estimators=self.n_estimators)
        rf.fit(X, y)
        predict_prob = rf.predict_proba(TEST)
        return predict_prob, lbl

    def train(self, X, shop_ids, TEST):
        """
        预测标签
        """
        lbl = preprocessing.LabelEncoder()
        lbl.fit(shop_ids)
        y = lbl.transform(shop_ids)
        rf = RandomForestClassifier(n_jobs=-1, n_estimators=self.n_estimators)
        rf.fit(X, y)
        predict = rf.predict(TEST)
        predict_ids = [lbl.inverse_transform(int(x)) for x in predict]
        return predict_ids
