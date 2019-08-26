# 统计 post 和 response 中单词的词频，选取前 VOCABULARY_NUM 个单词作为我们的词汇表
import json

data_set = []
with open('../data/train_set.txt') as f:
    for line in f:
        data_set.append(json.loads(line))

posts = []
responses = []
for items in data_set:
    posts.append(items['post'])
    responses.append(items['response'])

vocabulary = {}
for post in posts:
    for v in post:
        vocabulary.setdefault(v, 0)
        vocabulary[v] = vocabulary[v] + 1
for response in responses:
    for v in response:
        vocabulary.setdefault(v, 0)
        vocabulary[v] = vocabulary[v] + 1
print(len(vocabulary))

vocabulary = ['_PAD', '_UNK', '_GO', '_EOS'] + sorted(vocabulary, key=vocabulary.get, reverse=True)  # 将词汇表按照词频降序
with open('../data/vocabulary.txt', 'w') as f:
    for v in vocabulary:
        f.write(v + '\n')
