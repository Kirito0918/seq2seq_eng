import json
import numpy as np
import tensorflow as tf
from seq2seq import Seq2seq
import time
import random
random.seed(time.time())

TRAIN = True
TRAIN_CONTINUE = False  # 是否继续训练
TEST = False
INFERENCE = False
NUM_SYMBOL = 30000
NUM_UNITS = 128
OUTPUT_ALIGNMENTS = False
BATCH_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.0001
MAX_GRADIENT_NORM = 5.0
MAX_LENGTH = 60
TRAIN_DIR = './train'

# 补齐句子
def padding(str_list, length):
    str_padding = str_list + ['_EOS'] + ['_PAD' for _ in range(length-len(str_list)-1)]
    return str_padding

# 得到一批数据
def get_data(batch_data):
    # 构造 posts 和 responses
    posts = []
    responses = []
    for data in batch_data:
        posts.append(data['post'])
        responses.append(data['response'])

    # 获取 post 和 response 的长度列表，和编码器解码器长度
    posts_len = []
    for post in posts:
        posts_len.append(len(post) + 1)
    encoder_len = max(posts_len)
    responses_len = []
    for response in responses:
        responses_len.append(len(response) + 1)
    decoder_len = max(responses_len)

    # 补齐 post 和 response
    posts = np.array([padding(post, encoder_len) for post in posts])
    responses = np.array([padding(response, decoder_len) for response in responses])

    return {"posts_len": np.array(posts_len, dtype=np.int32),
            "responses_len": np.array(responses_len, dtype=np.int32),
            "posts_string": posts,
            "responses_string": responses}

# 在验证集上计算每个句子平均的损失
def evalueate(sess, model, validation_set):
    validation_set_len = len(validation_set)
    start = 0
    losses = np.zeros((1))  # 所有数据的损失之和
    while start < validation_set_len:
        end = start + BATCH_SIZE
        if end > validation_set_len:
            batch_data = get_data(validation_set[start:])
        else:
            batch_data = get_data(validation_set[start: end])
        start = end
        loss = model.test(sess, batch_data)  # 一个 batch 的数据的总损失
        losses = losses + np.sum(loss[0])
    return losses / validation_set_len

def main():
    # 读取训练集
    data_set = []
    with open('./data/train_set.txt') as f:
        for line in f:
            data_set.append(json.loads(line))
    print("num_data: %s" % len(data_set))

    # 读取验证集
    validation_set = []
    with open('./data/valid_set.txt') as f:
        for line in f:
            validation_set.append(json.loads(line))
    print("num_validation: %s" % len(validation_set))

    # 读取词汇表
    vocabulary = []
    with open('./data/vocabulary.txt') as f:
        for line in f:
            vocabulary.append(line.strip())
    vocabulary = vocabulary[: NUM_SYMBOL]
    print("num_symbol: %s" % len(vocabulary))

    # 载入词向量
    vectors = {}
    with open('./data/glove.840B.300d.txt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            word = line[:line.find(' ')]
            vector = line[line.find(' ') + 1:]
            vectors[word] = vector

    # 构造我们词汇表的词嵌入
    embed = []
    for word in vocabulary:
        if word in vectors:
            embed.append(list(map(float, vectors[word].split())))
        else:
            embed.append(np.zeros((300), dtype=np.float32))
    embed = np.array(embed, dtype=np.float32)
    print("embed_shape: ", np.shape(embed))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 允许 GPU 自动分配资源
    with tf.Session(config=config) as sess:
        model = Seq2seq(NUM_SYMBOL,
                        NUM_UNITS,
                        NUM_LAYERS,
                        embed,
                        max_length=MAX_LENGTH,
                        learning_rate=LEARNING_RATE,
                        max_gradient_norm=MAX_GRADIENT_NORM,
                        output_alignments=OUTPUT_ALIGNMENTS)
        tf.global_variables_initializer().run()

        op_in = model.symbol2index.insert(tf.constant(vocabulary, dtype=tf.string),
                                          tf.constant(list(range(NUM_SYMBOL)), dtype=tf.int64))
        sess.run(op_in)
        op_out = model.index2symbol.insert(tf.constant(list(range(NUM_SYMBOL)), dtype=tf.int64),
                                           tf.constant(vocabulary, dtype=tf.string))
        sess.run(op_out)

        if TRAIN:
            if TRAIN_CONTINUE:
                model.saver.restore(sess, tf.train.latest_checkpoint(TRAIN_DIR))
            print(model.print_parameters())
            epoch = 0
            while True:
                random.shuffle(data_set)
                start = 0
                while start < len(data_set):
                    if start >= len(data_set):
                        break
                    end = start + BATCH_SIZE
                    if end >= len(data_set):
                        batch_data = get_data(data_set[start:])
                    else:
                        batch_data = get_data(data_set[start: end])
                    _, loss, total_loss = model.train(sess, batch_data)
                    batch_loss = total_loss / len(batch_data)
                    if model.global_step.eval() % 20000 == 0:
                        print("epoch: %s" % epoch, end=" ")
                        print("global_step: %s" % model.global_step.eval(), end=" ")
                        print("start: %s" % start, end=" ")
                        print("loss = ", loss, end=" ")
                        print("ppl = ", np.exp(batch_loss))
                        model.saver.save(sess, '%s/checkpoint' % TRAIN_DIR, global_step=model.global_step)
                    start = end
                epoch = epoch + 1
                loss_per_data = evalueate(sess, model, validation_set)
                print("mean loss of per data on validation set: ", loss_per_data)
                print("ppl of per data on validation set: ", np.exp(loss_per_data))

        if TEST:
            model.saver.restore(sess, tf.train.latest_checkpoint(TRAIN_DIR))
            fr = open('./data/test_set.txt', 'r', encoding='utf8')
            fw = open('./data/result.txt', 'w', encoding='utf8')

            test_set = []
            for line in fr:
                data_set.append(json.loads(line))
            loss_per_data = evalueate(sess, model, test_set)
            ppl = "ppl of per data on validation set: %f" % np.exp(loss_per_data)
            fw.write(ppl)

            for line in fr:
                data = json.loads(line)
                post = data['post']
                posts_len = [len(post)]
                posts_string = [post]
                words = model.inference(sess, {"posts_len": np.array(posts_len, dtype=np.int32),
                                               "posts_string": np.array(posts_string)})
                word = words[0][0]
                word = [str(item, encoding="utf-8") for item in word]
                word = word[: word.find('_EOS')]
                data['result'] = word
                fw.write(json.dumps(data)+'\n')

            fw.close()
            fr.close()

        if INFERENCE:
            model.saver.restore(sess, tf.train.latest_checkpoint(TRAIN_DIR))
            while True:
                print("post: ", end=" ")
                post = input()
                post = post.split()
                posts_len = [len(post)]
                posts_string = [post]
                words = model.inference(sess, {"posts_len": np.array(posts_len, dtype=np.int32),
                                               "posts_string": np.array(posts_string)})
                word = words[0][0]
                word = [str(item, encoding="utf-8") for item in word]
                word = " ".join(word)
                word = word[: word.find('_EOS')]
                print("response: ", word)

if __name__ == '__main__':
    main()
