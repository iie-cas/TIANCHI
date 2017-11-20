# coding: utf-8
import os

def get_result(dir_name, file_name):
    """
    dir_name: 文件夹名称
    file_name: 结果文件名
    """
    file_list = os.listdir(dir_name)
    fw = open(file_name, 'w')
    fw.write('row_id,shop_id\n')
    for file_name in file_list:
        if 'm_' in file_name:
            with open(dir_name + file_name, 'r') as f:
                for line in f.readlines()[1:]:
                    if line.strip() != '':
                        fw.write(line)
    fw.close()

if __name__ == '__main__':
    # 获得随机森林的预测结果
    get_result('./rf_result/', 'rf_result.csv')
    # 获得XGBoost的预测结果
    get_result('./xgb_result/', 'xgb_result.csv')