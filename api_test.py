import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell

######################################tf.nn.dynamic_rnn测试
# 1 维的词嵌入，共有 10 个词
embedding = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=np.float32)
post = np.array([[5, 2, 3, 0], [9, 5, 0, 0]], dtype=np.int32)  # 样本 batch_size=2 , encoder_len=4
length = np.array([3, 2], dtype=np.int32)

embedding = tf.get_variable(name="embedding", initializer=embedding, dtype=tf.float32)
post = tf.get_variable(name="post", initializer=post, dtype=tf.int32)
post = tf.nn.embedding_lookup(embedding, post)

encoder_cell = MultiRNNCell([LSTMCell(32) for _ in range(3)])  # 编码器为 3 层 LSTM，LSTM 的单元个数为 32

encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, post, length, dtype=tf.float32, scope="encoder")

# 编码器输入为 [2, 4, 1] batch_size=2 , encoder_len=4，embedding_size=1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(encoder_output))  # 超过句子长度的部分输出全为 0
    print(sess.run(tf.shape(encoder_output)))  # [2, 4, 32]

    # print(sess.run(encoder_state))
    # print(sess.run(tf.shape(encoder_state)))  # [3, 2, 2, 32]=[layers, (c,h), batch, units]

    # print(sess.run(encoder_state[-1]))
    # print(sess.run(tf.shape(encoder_state[-1])))  # [2, 2, 32]=[(c,h), batch, units]

    print(sess.run(encoder_state[-1].h))  # h 为输出不为 0 前的最后一次输出，c 为细胞状态
    print(sess.run(tf.shape(encoder_state[-1].h)))  # [2, 32]=[batch, units]
####################################################
# print(model.print_parameters())
# print("encoder rnn_cell0's weight:", sess.run(['encoder/multi_rnn_cell/cell_0/lstm_cell/kernel:0']))
# print("encoder rnn_cell0's bias:", sess.run(['encoder/multi_rnn_cell/cell_0/lstm_cell/bias:0']))
# print("encoder rnn_cell1's weight:", sess.run(['encoder/multi_rnn_cell/cell_1/lstm_cell/kernel:0']))
# print("encoder rnn_cell1's bias:", sess.run(['encoder/multi_rnn_cell/cell_1/lstm_cell/bias:0']))
#
# print("decoder rnn_cell0's weight:",
#       sess.run(['decoder/decoder_rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0']))
# print("decoder rnn_cell0's bias:", sess.run(['decoder/decoder_rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0']))
# print("decoder rnn_cell1's weight:",
#       sess.run(['decoder/decoder_rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0']))
# print("decoder rnn_cell1's bias:", sess.run(['decoder/decoder_rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0']))
#
# print("keys weight:", sess.run(['decoder/attention_keys/weights:0']))
# print("score attnW:", sess.run(['decoder/attention_score/attnW:0']))
# print("score attnV:", sess.run(['decoder/attention_score/attnV:0']))
# print("construct weight:", sess.run(['decoder/attention_construct/weights:0']))
#
# print("output weight:", sess.run(['decoder/decoder_rnn/output_projection/weights:0']))
# print("output bias", sess.run(['decoder/decoder_rnn/output_projection/biases:0']))

