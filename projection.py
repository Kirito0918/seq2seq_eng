import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers

def output_projection_layer(num_units,  # decoder_num_units
                            num_symbols,  # 词汇表大小
                            num_samples=None,
                            name="output_projection"):
    def output_fn(outputs):  # [batch_size, decoder_len, num_units]
        return layers.linear(outputs, num_symbols, scope=name)  # [batch_size, decoder_len, num_symbols]

    def sequence_loss(outputs,  # 解码器输出
                      targets,  # 标签
                      masks):  # 对标签的 mask
        with tf.variable_scope('decoder_rnn'):
            # 预测值
            logits = layers.linear(outputs, num_symbols, scope=name)  # [batch_size, decoder_len, num_symbols]
            logits = tf.reshape(logits, [-1, num_symbols])  # [batch_size*decoder_len, num_symbols]
            # 标签
            local_labels = tf.reshape(targets, [-1])  # [batch_size*decoder_len]
            local_masks = tf.reshape(masks, [-1])  # [batch_size*decoder_len]
            # 计算损失
            local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=local_labels, logits=logits)  # [batch_size*decoder_len]
            # 序列长度外的部分不计算损失
            local_loss = local_loss * local_masks
            loss = tf.reduce_sum(local_loss)  # 序列的总损失 [batch_size*decoder_len]
            total_size = tf.reduce_sum(local_masks)  # 序列的总长度
            total_size += 1e-12  # 避免总长度为0
            return loss, loss / total_size  # 一个 batch 的数据的每个单词的平均损失

    return output_fn, sequence_loss
