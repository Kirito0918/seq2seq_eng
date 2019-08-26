# 从别人的数据集中构建自己的数据集，每行为一个 json 对象，内容为{"post":[], "response":[]}
import json

DATA_NUM = 1200000  # 训练集个数
# VALIDATION_NUM = 20000  # 验证集个数

train_set = []
with open('../data/trainset.txt') as f:
    for idx, line in enumerate(f):
        if idx == DATA_NUM:
            break
        if idx % 10000 == 0:
            print('read train file line %d' % idx)
        train_set.append(json.loads(line))

valid_set = []
with open('../data/validset.txt') as f:
    for idx, line in enumerate(f):
        if idx % 10000 == 0:
            print('read valid file line %d' % idx)
        valid_set.append(json.loads(line))

test_set = []
with open('../data/testset.txt') as f:
    for idx, line in enumerate(f):
        if idx % 10000 == 0:
            print('read test file line %d' % idx)
        test_set.append(json.loads(line))

print("len(train_set): ", len(train_set))
print("len(valid_set): ", len(valid_set))
print("len(test_set): ", len(test_set))

with open('../data/train_set.txt', 'w') as f:
    for items in train_set:
        data = {}
        data['post'] = items['post']
        data['response'] = items['response']
        f.write(json.dumps(data) + '\n')

with open('../data/valid_set.txt', 'w') as f:
    for items in valid_set:
        data = {}
        data['post'] = items['post']
        data['response'] = items['response']
        f.write(json.dumps(data) + '\n')

with open('../data/test_set.txt', 'w') as f:
    for items in test_set:
        data = {}
        data['post'] = items['post']
        data['response'] = items['response']
        f.write(json.dumps(data) + '\n')
