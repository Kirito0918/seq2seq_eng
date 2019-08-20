import numpy as np
import tensorflow as tf
import json
from tensorflow.python.framework import constant_op
import sys
import math
import os
import time
import random
random.seed(time.time())
from model import Model, _START_VOCAB

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")  # 是否训练
tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")  # 词汇表size
tf.app.flags.DEFINE_integer("num_entities", 21471, "entitiy vocabulary size.")  # 实体词汇size
tf.app.flags.DEFINE_integer("num_relations", 44, "relation size.")  # 关系size
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")  # 词嵌入size
tf.app.flags.DEFINE_integer("trans_units", 100, "Size of trans embedding.")  # trans嵌入size
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")  # 每层的size
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")  # 层数
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")  # batch_size
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")  # 数据的目录
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")  # 保存模型的目录
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")  # 每多少步保存一下模型
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")  # 推导的版本
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")  # 是否显示参数
tf.app.flags.DEFINE_string("inference_path", "test", "Set filename of inference")  # 推导的文件名

# 保存命令行参数
FLAGS = tf.app.flags.FLAGS

# 如果那个目录的最后一个字符为'/'，则去掉该字符
if FLAGS.train_dir[-1] == '/':
    FLAGS.train_dir = FLAGS.train_dir[:-1]

csk_triples, csk_entities, kb_dict = [], [], []

def prepare_data(path, is_train=True):
    """
    准备数据
    is_train: 是否训练，不训练就不载入训练集
    """
    global csk_entities, csk_triples, kb_dict
    
    with open('%s/resource.txt' % path) as f:
        d = json.loads(f.readline())

    csk_triples = d['csk_triples']  # 三元组列表 ["实体,关系,实体",...]
    csk_entities = d['csk_entities']  # 实体列表 ["实体",...]
    raw_vocab = d['vocab_dict']  # 词汇表，是一个字典
    kb_dict = d['dict_csk']  # 知识图字典 {"实体"：["实体,关系,实体" ,...}

    data_train, data_dev, data_test = [], [], []

    # 载入训练集
    if is_train:
        with open('%s/trainset.txt' % path) as f:
            for idx, line in enumerate(f):
                if idx % 100000 == 0:
                    print('read train file line %d' % idx)
                data_train.append(json.loads(line))
                if idx == 100000:  # 用来删减数据集
                    break

    # 载入验证集
    with open('%s/validset.txt' % path) as f:
        for line in f:
            data_dev.append(json.loads(line))

    # 载入测试集
    with open('%s/testset.txt' % path) as f:
        for line in f:
            data_test.append(json.loads(line))

    return raw_vocab, data_train, data_dev, data_test

def build_vocab(path, raw_vocab, trans='transE'):
    """
    构建词汇表
    """
    # 创建词汇表
    print("Creating word vocabulary...")
    # _START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']，将这些词加入词汇表头部，并将词汇表排序
    # sorted()将原始的词汇表排序，排序的key为字典value，reverse=True为降序
    # value 应该是词频
    vocab_list = _START_VOCAB + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    # 如果词汇表大于我们设置的词汇表size，则只截取前 symbols 个词汇
    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]

    # 创建实体词汇表
    # entity.txt 内容的格式为每行一个实体单词
    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    with open('%s/entity.txt' % path) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)

    # 创建关系词汇表
    # relation.txt 的内容格式为每行一个关系单词
    print("Creating relation vocabulary...")
    relation_list = []
    with open('%s/relation.txt' % path) as f:
        for i, line in enumerate(f):
            r = line.strip()
            relation_list.append(r)

    # 载入词向量
    # glove.840B.300d.txt 的内容格式为每行为 单词 空格 向量
    print("Loading word vectors...")
    vectors = {}  # 词向量字典
    with open('%s/glove.840B.300d.txt' % path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    embed = []  # 词嵌入列表
    for word in vocab_list:  # 对于词汇表中的每个单词
        if word in vectors:  # 如果单词在词向量的字典中
            vector = list(map(float, vectors[word].split()))  # 将词向量的字符串分割后转为 float 类型
        else:  # 如果单词不在词向量的字典中
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)  # 将向量初始化为全1向量
        embed.append(vector)  # 将词向量加入到词嵌入列表
    embed = np.array(embed, dtype=np.float32)

    # 载入实体向量
    # entity_%s.txt 的内容是每行一个向量
    print("Loading entity vectors...")
    entity_embed = []
    with open('%s/entity_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            entity_embed.append(list(map(float, s)))

    # 载入关系向量
    # relation_%s.txt 的内容是每行一个向量
    print("Loading relation vectors...")
    relation_embed = []
    with open('%s/relation_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            relation_embed.append(s)

    # 将实体嵌入和关系嵌入拼接成一个数组
    entity_relation_embed = np.array(entity_embed+relation_embed, dtype=np.float32)
    entity_embed = np.array(entity_embed, dtype=np.float32)
    relation_embed = np.array(relation_embed, dtype=np.float32)

    return vocab_list, embed, entity_list, entity_embed, relation_list, relation_embed, entity_relation_embed

def gen_batched_data(data):
    """
    格式化一批数据
    data: 数据集
    """
    global csk_entities, csk_triples, kb_dict  # 实体列表，三元组列表，知识图字典
    encoder_len = max([len(item['post']) for item in data])+1  # 所有 post 长度的最大值 +1
    decoder_len = max([len(item['response']) for item in data])+1  # 所有 response 长度的最大值 +1
    triple_num = max([len(item['all_triples']) for item in data])+1  # 一个 post 检索到的知识图最大数量 +1
    triple_len = max([len(tri) for item in data for tri in item['all_triples']])  # 知识图中包含三元组个数的最大值
    max_length = 20

    posts, responses, posts_length, responses_length = [], [], [], []

    entities, triples, matches, post_triples, response_triples = [], [], [], [], []
    match_entities, all_entities = [], []

    match_triples, all_triples = [], []
    NAF = ['_NAF_H', '_NAF_R', '_NAF_T']

    # 给句子补全长度，句子结束添加['_EOS']，后面小于 encoder/decoder_len 的部分补上 _PAD
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

    def padding_triple(triple, num, l):
        """
        把每个关键字的知识图补成，关键词数量相同，每个关键词的关系三元组数量相同
        triple: 一个 post 的检索到的所有知识图
        num: =triple_num 所有 all_triples 包含 tri 个数的最大值 +1
        l: =triple_len all_triple 中 tri 的最大长度
        return: 补全后的三元组
        """
        newtriple = []
        triple = [[NAF]] + triple  # 将[[['_NAF_H', '_NAF_R', '_NAF_T']]] 加入 triple
        # 将 all_triples 中每一项所指的三元组个数都补成相同，为 triple_len
        for tri in triple:
            newtriple.append(tri + [['_PAD_H', '_PAD_R', '_PAD_T']] * (l-len(tri)))
        pad_triple = [['_PAD_H', '_PAD_R', '_PAD_T']] * l
        # 将每个 all_triples 所包含的 triple 项数都补成相同，为 triple_len
        return newtriple + [pad_triple] * (num - len(newtriple))

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)  # post 长度的列表，长度包含了 _EOS，所以 +1
        responses_length.append(len(item['response'])+1)  # response 长度的列表，长度包含了 _EOS，所以 +1
        # 把每个关键字的知识图补成，关键词数量相同，每个关键词的关系三元组数量相同
        all_triples.append(padding_triple([[csk_triples[x].split(', ') for x in triple] for triple in item['all_triples']], triple_num, triple_len))
        # 将 post_triples 补齐到 encoder_len 长度
        post_triples.append([[x] for x in item['post_triples']] + [[0]] * (encoder_len - len(item['post_triples'])))
        # response_triples 为[[NAF],..., [['实体'，'关系'，'实体']], [NAF]...]格式，不为-1就用三元组替换，最后将其补齐到 decoder_len 长度
        response_triples.append([NAF] + [NAF if x == -1 else csk_triples[x].split(', ') for x in item['response_triples']] + [NAF] * (decoder_len - 1 - len(item['response_triples'])))
        # 回复用到的三元组在 all_triples 中的 index
        match_index = []
        for idx, x in enumerate(item['match_index']):
            _index = [-1] * triple_num
            if x[0] == -1 and x[1] == -1:
                match_index.append(_index)
            else:
                _index[x[0]] = x[1]
                t = all_triples[-1][x[0]][x[1]]
                assert(t == response_triples[-1][idx+1])
                match_index.append(_index)
        # decoder_len 表示回复的哪个时间步用了知识三元组，最后一维的下标表示用了第几个图，值表示用了该图第几个关系
        match_triples.append(match_index + [[-1]*triple_num]*(decoder_len-len(match_index)))  # batch_size * decoder_len * triple_num

        # 如果不在训练，构建entities
        # entities 为 post 中每个关键词用到的关系三元组中所有实体
        if not FLAGS.is_train:
            entity = [['_NONE']*triple_len]
            for ent in item['all_entities']:
                entity.append([csk_entities[x] for x in ent] + ['_NONE'] * (triple_len-len(ent)))
            entities.append(entity+[['_NONE']*triple_len]*(triple_num-len(entity)))


    batched_data = {'posts': np.array(posts),
            'responses': np.array(responses),
            'posts_length': posts_length, 
            'responses_length': responses_length,
            'triples': np.array(all_triples),
            'entities': np.array(entities),
            'posts_triple': np.array(post_triples),
            'responses_triple': np.array(response_triples),
            'match_triples': np.array(match_triples)}

    return batched_data

# 训练模型
def train(model, sess, data_train):
    batched_data = gen_batched_data(data_train)
    outputs = model.step_decoder(sess, batched_data)
    return np.sum(outputs[0])  # 一个 batch 损失之和

def generate_summary(model, sess, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(selected_data)
    summary = model.step_decoder(sess, batched_data, forward_only=True, summary=True)[-1]
    return summary

# 验证集上计算 ppx，这部分代码没什么好看的
def evaluate(model, sess, data_dev, summary_writer):
    loss = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True)
        loss += np.sum(outputs[0])
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= len(data_dev)
    summary = tf.Summary()
    summary.value.add(tag='decoder_loss/dev', simple_value=loss)
    summary.value.add(tag='perplexity/dev', simple_value=np.exp(loss))
    summary_writer.add_summary(summary, model.global_step.eval())
    print('    perplexity on dev set: %.2f' % np.exp(loss))

# 返回记录模型的几个 globe_step
def get_steps(train_dir):
    a = os.walk(train_dir)
    for root, dirs, files in a:
        if root == train_dir:
            filenames = files

    steps, metafiles, datafiles, indexfiles = [], [], [], []
    for filename in filenames:
        if 'meta' in filename:
            metafiles.append(filename)
        if 'data' in filename:
            datafiles.append(filename)
        if 'index' in filename:
            indexfiles.append(filename)

    metafiles.sort()
    datafiles.sort()
    indexfiles.sort(reverse=True)

    for f in indexfiles:
        steps.append(int(f[11:-6]))

    return steps

def test(sess, saver, data_dev, setnum=5000):
    # 载入 stopwords
    with open('%s/stopwords' % FLAGS.data_dir) as f:
        stopwords = json.loads(f.readline())
    # 获得记录模型的几个 globe_step
    steps = get_steps(FLAGS.train_dir)
    low_step = 00000
    high_step = 800000
    with open('%s.res' % FLAGS.inference_path, 'w') as resfile, open('%s.log' % FLAGS.inference_path, 'w') as outfile:
        # 对最近保存的几次模型都要进行一次下面的操作
        for step in [step for step in steps if step > low_step and step < high_step]:
            outfile.write('test for model-%d\n' % step)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, step)
            print('restore from %s' % model_path)
            try:
                saver.restore(sess, model_path)
            except:
                continue
            st, ed = 0, FLAGS.batch_size
            results = []
            loss = []
            while st < len(data_dev):
                selected_data = data_dev[st:ed]
                batched_data = gen_batched_data(selected_data)
                responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'],
                    {'enc_inps:0': batched_data['posts'],
                     'enc_lens:0': batched_data['posts_length'],
                     'dec_inps:0': batched_data['responses'],
                     'dec_lens:0': batched_data['responses_length'],
                     'entities:0': batched_data['entities'],
                     'triples:0': batched_data['triples'],
                     'match_triples:0': batched_data['match_triples'],
                     'enc_triples:0': batched_data['posts_triple'],
                     'dec_triples:0': batched_data['responses_triple']})
                loss += [x for x in ppx_loss]
                for response in responses:
                    result = []
                    for token in response:
                        if token != '_EOS':
                            result.append(token)
                        else:
                            break
                    results.append(result)
                st, ed = ed, ed+FLAGS.batch_size
            match_entity_sum = [.0] * 4
            cnt = 0
            for post, response, result, match_triples, triples, entities in zip([data['post'] for data in data_dev], [data['response'] for data in data_dev], results, [data['match_triples'] for data in data_dev], [data['all_triples'] for data in data_dev], [data['all_entities'] for data in data_dev]):
                #
                setidx = int(cnt / setnum)
                result_matched_entities = []
                triples = [csk_triples[tri] for triple in triples for tri in triple]
                match_triples = [csk_triples[triple] for triple in match_triples]
                entities = [csk_entities[x] for entity in entities for x in entity]
                matches = [x for triple in match_triples for x in [triple.split(', ')[0], triple.split(', ')[2]] if x in response]
                
                for word in result:
                    if word not in stopwords and word in entities:
                        result_matched_entities.append(word)
                #
                _result = [str(res) for res in result]
                outfile.write('post: %s\nresponse: %s\nresult: %s\nmatch_entity: %s\n\n' % (' '.join(post), ' '.join(response), ' '.join(_result), ' '.join(result_matched_entities)))
                match_entity_sum[setidx] += len(set(result_matched_entities))
                cnt += 1
            match_entity_sum = [m / setnum for m in match_entity_sum] + [sum(match_entity_sum) / len(data_dev)]
            losses = [np.sum(loss[x:x+setnum]) / float(setnum) for x in range(0, setnum*4, setnum)] + [np.sum(loss) / float(setnum*4)]
            losses = [np.exp(x) for x in losses]
            def show(x):
                return ', '.join([str(v) for v in x])
            outfile.write('model: %d\n\tperplexity: %s\n\tmatch_entity_rate: %s\n%s\n\n' % (step, show(losses), show(match_entity_sum), '='*50))
            resfile.write('model: %d\n\tperplexity: %s\n\tmatch_entity_rate: %s\n\n' % (step, show(losses), show(match_entity_sum)))
            outfile.flush()
            resfile.flush()
    return results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 允许 GPU 自动分配资源
with tf.Session(config=config) as sess:
    if FLAGS.is_train:  # 如果在训练
        # 载入原始词汇表，训练集，验证集和测试集
        raw_vocab, data_train, data_dev, data_test = prepare_data(FLAGS.data_dir)
        # 构建词汇表，词嵌入，实体表，实体嵌入，关系表，关系嵌入，实体+关系嵌入
        vocab, embed, entity_vocab, entity_embed, relation_vocab, relation_embed, entity_relation_embed = build_vocab(FLAGS.data_dir, raw_vocab)
        FLAGS.num_entities = len(entity_vocab)  # 实体的个数
        print(FLAGS.__flags)  # 输出命令行参数
        # 创建模型
        model = Model(
                FLAGS.symbols,  # 词汇表size
                FLAGS.embed_units,  # 词嵌入size
                FLAGS.units,  # 每层size
                FLAGS.layers,  # 层数
                embed,  # 传入的词嵌入
                entity_relation_embed,  # 传入的实体+关系嵌入
                num_entities=len(entity_vocab)+len(relation_vocab),  # 实体个数 = 实体数量+关系数量
                num_trans_units=FLAGS.trans_units)  # 实体关系嵌入的size
        # 如果保存了模型，就载入参数
        if tf.train.get_chectkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            # 给字符转 index 的 hash 表初始化
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)
            # 给 index 转字符的 hash 表初始化
            op_out = model.index2symbol.insert(constant_op.constant(
                list(range(FLAGS.symbols)), dtype=tf.int64), constant_op.constant(vocab))
            sess.run(op_out)
            # 初始化
            op_in = model.entity2index.insert(constant_op.constant(entity_vocab+relation_vocab),
                constant_op.constant(list(range(len(entity_vocab)+len(relation_vocab))), dtype=tf.int64))
            sess.run(op_in)
            # 初始化
            op_out = model.index2entity.insert(constant_op.constant(
                list(range(len(entity_vocab)+len(relation_vocab))), dtype=tf.int64), constant_op.constant(entity_vocab+relation_vocab))
            sess.run(op_out)

        # 如果显示参数，则输出参数
        if FLAGS.log_parameters:
            model.print_parameters()

        # 向该文件保存模型
        summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
        loss_step, time_step = np.zeros((1, )), .0  # [0.0]，0.0
        previous_losses = [1e18]*3  # [1e+18, 1e+18, 1e+18]
        train_len = len(data_train)  # 训练集的长度
        while True:
            st, ed = 0, FLAGS.batch_size * FLAGS.per_checkpoint
            # 将训练集随机排序
            random.shuffle(data_train)
            # 当 start 小于训练集的长度
            while st < train_len:
                start_time = time.time()  # 记录一下开始的时间
                # 循环 per_checkpoint 次
                for batch in range(st, ed, FLAGS.batch_size):
                    loss_step += train(model, sess, data_train[batch:batch+FLAGS.batch_size]) / (ed - st)  # 总感觉这条代码会出数组越界的错误
                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                # 输出模型总的更新参数的次数，学习速率，更新一次参数所花的时间，记录一次模型时的 ppx
                print("global step %d learning rate %.4f step-time %.2f loss %f perplexity %s"
                        % (model.global_step.eval(), model.lr, 
                            (time.time() - start_time) / ((ed - st) / FLAGS.batch_size), loss_step, show(np.exp(loss_step))))
                # 保存模型
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, 
                        global_step=model.global_step)
                summary = tf.Summary()
                summary.value.add(tag='decoder_loss/train', simple_value=loss_step)
                summary.value.add(tag='perplexity/train', simple_value=np.exp(loss_step))
                summary_writer.add_summary(summary, model.global_step.eval())
                summary_model = generate_summary(model, sess, data_train)
                summary_writer.add_summary(summary_model, model.global_step.eval())

                evaluate(model, sess, data_dev, summary_writer)
                previous_losses = previous_losses[1:]+[np.sum(loss_step)]
                loss_step, time_step = np.zeros((1, )), .0
                st, ed = ed, min(train_len, ed + FLAGS.batch_size * FLAGS.per_checkpoint)
            model.saver_epoch.save(sess, '%s/epoch/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
    else:  # 测试
        model = Model(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                embed=None,
                num_entities=FLAGS.num_entities+FLAGS.num_relations,
                num_trans_units=FLAGS.trans_units)

        if FLAGS.inference_version == 0:  # 载入最新的模型
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:  # 选择一个载入
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)
        saver = model.saver

        raw_vocab, data_train, data_dev, data_test = prepare_data(FLAGS.data_dir, is_train=False)

        test(sess, saver, data_test, setnum=5000)

