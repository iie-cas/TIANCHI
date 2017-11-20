from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class RF:
    """
    Random Forest
    """
    def __init__(self):
        self.n_estimators_options = [100, 120, 140, 160, 180, 200]
        self.best_n_estimators = 0
        self.best_acc = 0

    def train(self, mall_id, X, shop_ids, TEST, row_ids):
        """
        mall_id: 商场 ID
        X: 训练集 vector
        shop_ids: 训练集标签
        TEST： 测试集 vector
        row_ids: 测试集行号
        """
        lbl = preprocessing.LabelEncoder()
        lbl.fit(shop_ids)
        y = lbl.transform(shop_ids)
        # 划分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # 简单寻参
        for n_estimators_size in self.n_estimators_options:
            alg = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators_size)
            alg.fit(X_train, y_train)
            predict = alg.predict(X_test)
            acc = (y_test == predict).mean()
            print(n_estimators_size, acc)
            if acc >= self.best_acc:
                self.best_acc = acc
                self.best_n_estimators = n_estimators_size
        # 定义模型，训练
        rf = RandomForestClassifier(n_jobs=-1, n_estimators=self.best_n_estimators)
        rf.fit(X, y)
        predict = rf.predict(TEST)
        predict_result = [lbl.inverse_transform(int(x)) for x in predict]
        with open('./rf_result/' + str(mall_id) + '_result.csv', 'w') as f:
            f.write('row_id,shop_id\n')
            for i, row_id in enumerate(row_ids):
                f.write('%s,%s\n' %(row_id, predict_result[i]))
