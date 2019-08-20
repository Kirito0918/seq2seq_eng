# 从别人的数据集中构建自己的数据集，每行为一个 json 对象，内容为{"post":[], "response":[]}
import json

DATA_NUM = 1000000  # 训练集个数
VALIDATION_NUM = 20000  # 验证集个数

data_set = []
total_num = DATA_NUM + VALIDATION_NUM
with open('../data/trainset.txt') as f:
    for idx, line in enumerate(f):
        if idx == total_num:
            break
        if idx % 10000 == 0:
            print('read train file line %d' % idx)
        data_set.append(json.loads(line))
train_set = data_set[: DATA_NUM]
validation_set = data_set[DATA_NUM:]

print("len(data_set): ", len(data_set))
print("len(train_set): ", len(train_set))
print("len(validation_set): ", len(validation_set))

with open('../data/dataset.txt', 'w') as f:
    for items in train_set:
        data = {}
        data['post'] = items['post']
        data['response'] = items['response']
        f.write(json.dumps(data) + '\n')

with open('../data/validationset.txt', 'w') as f:
    for items in validation_set:
        data = {}
        data['post'] = items['post']
        data['response'] = items['response']
        f.write(json.dumps(data) + '\n')
