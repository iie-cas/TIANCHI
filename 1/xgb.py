import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def analyse(mall_id, X, shop_ids):
    """
    划分训练集和验证集，计算 ACC
    """
    lbl = preprocessing.LabelEncoder()
    lbl.fit(shop_ids)
    y = lbl.transform(shop_ids)
    # 计算类别数
    num_class = y.max() + 1
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    # 定义参数
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': num_class,
        'silent': 1,
    }
    bst = xgb.train(params, xg_train, 60, watchlist, early_stopping_rounds=15)
    pred = bst.predict(xg_test)
    acc = (y_test == pred).mean()
    print('accuracy: %s' %acc)


def train(mall_id, X, shop_ids, TEST, row_ids):
    """
    训练预测
    """
    lbl = preprocessing.LabelEncoder()
    lbl.fit(shop_ids)
    y = lbl.transform(shop_ids)
    num_class = y.max() + 1
    xg_train = xgb.DMatrix(X, label=y)
    xg_test = xgb.DMatrix(TEST)
    watchlist = [(xg_train, 'train')]
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': num_class,
        'silent': 1,
    }
    bst = xgb.train(params, xg_train, 60, watchlist, early_stopping_rounds=15)
    pred = bst.predict(xg_test)
    pred = [lbl.inverse_transform(int(x)) for x in pred]
    # 写出结果到文件
    with open('./xgb_result/' + str(mall_id) + '_result.csv', 'w') as f:
        f.write('row_id,shop_id\n')
        for i, row_id in enumerate(row_ids):
            f.write('%s,%s\n' %(row_id, pred[i]))
