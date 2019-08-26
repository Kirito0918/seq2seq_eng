import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from projection import output_projection_layer
from attention_decoder import prepare_attention
from attention_decoder import attention_decoder_fn_train
from attention_decoder import attention_decoder_fn_inference
from decoder import dynamic_rnn_decoder


PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3

class Seq2seq(object):
    def __init__(self,
                 num_symbol,  # 词汇表大小
                 num_units,  # 隐藏层维度
                 num_layers,  # 编码/解码器层数
                 embed,  # 词嵌入
                 max_length=60,
                 learning_rate=0.0001,
                 max_gradient_norm=5.0,
                 output_alignments=False):  # 是否保存注意力权重
        # 词汇映射到 index 的 hash table
        self.symbol2index = MutableHashTable(key_dtype=tf.string,
                                        value_dtype=tf.int64,
                                        default_value=UNK_ID,
                                        shared_name="in_table",
                                        name="in_table",
                                        checkpoint=True)
        # index 映射到词汇的 hash table
        self.index2symbol = MutableHashTable(key_dtype=tf.int64,
                                        value_dtype=tf.string,
                                        default_value='_UNK',
                                        shared_name="out_table",
                                        name="out_table",
                                        checkpoint=True)

        # 模型变量
        self.posts_string = tf.placeholder(name="posts_string", shape=(None, None), dtype=tf.string)  # [batch_size, encoder_len]
        self.posts_len = tf.placeholder(name="posts_len", shape=(None), dtype=tf.int32)  # [batch_size]
        self.responses_string = tf.placeholder(name="responses_string", shape=(None, None), dtype=tf.string)  # [batch_size, decoder_len]
        self.responses_len = tf.placeholder(name="responses_len", shape=(None), dtype=tf.int32)  # [batch_size]
        self.embed = tf.get_variable("word_embed", dtype=tf.float32, initializer=embed)

        batch_size, encoder_len = tf.unstack(tf.shape(self.posts_string))
        decoder_len = tf.shape(self.responses_string)[1]

        # posts 和 responses 的序列表示
        self.posts_index = self.symbol2index.lookup(self.posts_string)  # [batch_size, encoder_len]
        self.responses_index = self.symbol2index.lookup(self.responses_string)  # [batch_size, decoder_len]

        # decoder 输入的序列表示
        self.responses_input_index = tf.concat([
            tf.ones((batch_size, 1), dtype=tf.int64) * GO_ID,
            tf.split(self.responses_index, [decoder_len-1, 1], axis=1)[0]
        ], axis=1)

        # encoder 和 decoder 的输入
        self.encoder_input = tf.nn.embedding_lookup(embed, self.posts_index)  # [batch_size, encoder_len, embedding_size]
        # decoder_label = tf.nn.embedding_lookup(embed, responses_index)  # [batch_size, decoder_len, embedding_size]
        self.decoder_input = tf.nn.embedding_lookup(embed, self.responses_input_index)  # [batch_size, decoder_len, embedding_size]

        self.decoder_mask = tf.reshape(
            tf.cumsum(
                tf.one_hot(self.responses_len - 1, decoder_len),
                reverse=True, axis=1
            ),
        [-1, decoder_len])  # [batch_size, decoder_len]

        encoder_cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        decoder_cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])

        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, self.posts_len, dtype=tf.float32, scope="encoder")

        output_fn, sequence_loss = output_projection_layer(num_units, num_symbol)

        # 训练
        with tf.variable_scope("decoder"):
            keys, values, attention_score_fn, attention_construct_fn = \
                prepare_attention(encoder_output, num_units, reuse=False)
            decoder_fn_train = attention_decoder_fn_train(encoder_state,
                                                          keys,
                                                          values,
                                                          attention_score_fn,
                                                          attention_construct_fn,
                                                          output_alignments=output_alignments,
                                                          decoder_len=decoder_len)
            self.decoder_output, _, alignments_ta = dynamic_rnn_decoder(decoder_cell,
                                                                        decoder_fn_train,
                                                                        inputs=self.decoder_input,
                                                                        sequence_length=self.responses_len,
                                                                        scope="decoder_rnn")
            self.total_loss, self.loss = sequence_loss(self.decoder_output, self.responses_index, self.decoder_mask)

        # 推导
        with tf.variable_scope("decoder", reuse=True):
            # 得到注意力函数
            keys, values, attention_score_fn, attention_construct_fn = \
                prepare_attention(encoder_output, num_units, reuse=True)
            decoder_fn_inference = attention_decoder_fn_inference(output_fn,
                                                                  encoder_state,
                                                                  keys,
                                                                  values,
                                                                  attention_score_fn,
                                                                  attention_construct_fn,
                                                                  self.embed,
                                                                  GO_ID,
                                                                  EOS_ID,
                                                                  max_length,
                                                                  num_symbol)
            # decoder_distribution: [batch_size, decoder_len, num_symbol] 未 softmax 的预测分布
            # output_ids_ta: decoder_len [bath_size]
            self.decoder_distribution, _, output_ids_ta = dynamic_rnn_decoder(decoder_cell,
                                                                              decoder_fn_inference,
                                                                              scope="decoder_rnn")
            # self.word_ids = tf.cast(tf.argmax(tf.nn.softmax(self.decoder_distribution), 2), dtype=tf.int64)
            # self.output_ids = tf.transpose(output_ids_ta.stack())

            output_len = tf.shape(self.decoder_distribution)[1]  # decoder_len
            self.output_ids = tf.transpose(output_ids_ta.gather(tf.range(output_len)))  # [batch_size, decoder_len]

            # 对 output 的值域行裁剪
            self.word_ids = tf.cast(tf.clip_by_value(self.output_ids, 0, num_symbol), tf.int64)  # [batch_size, decoder_len]
            self.words = self.index2symbol.lookup(self.word_ids)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.params = tf.global_variables()
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(gradients, self.params), global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=3)

    def train(self, sess, data):
        output = [self.update, self.loss, self.total_loss]

        input = {self.posts_len: data['posts_len'],
                 self.responses_len: data['responses_len'],
                 self.posts_string: data['posts_string'],
                 self.responses_string: data['responses_string']}

        return sess.run(output, input)

    def test(self, sess, data):
        output = [self.total_loss]

        input = {self.posts_len: data['posts_len'],
                 self.responses_len: data['responses_len'],
                 self.posts_string: data['posts_string'],
                 self.responses_string: data['responses_string']}

        return sess.run(output, input)

    def inference(self, sess, data):
        output = [self.words]

        input = {self.posts_len: data['posts_len'],
                 self.posts_string: data['posts_string']}

        return sess.run(output, input)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))


