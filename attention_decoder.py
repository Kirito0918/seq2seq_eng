import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import rnn_cell_impl

def attention_decoder_fn_train(encoder_state,  # 编码器输出状态 [layers, (c,h), batch, units]
                               keys,  # 注意力的 keys
                               values,  # 注意力的 values
                               attention_score_fn,
                               attention_construct_fn,
                               output_alignments=False,  # 是否输出注意力系数
                               decoder_len=None,  # 解码器最大长度
                               name=None):
    """返回一个训练时用到的注意力解码器函数
    用来在每一个时间步中，对输出进行处理，加上 attention，成为下一次的输入
    """
    with tf.name_scope(name, "attention_decoder_fn_train",
                       [encoder_state, keys, values, attention_score_fn, attention_construct_fn]):
        pass

    def decoder_fn(time,  # 当前将要进行的时间步
                   cell_state,  # RNN 长时记忆，初始化时需要赋值
                   cell_input,  # 当前时间步的输入
                   cell_output,  # 上一步的输出
                   context_state):  # 存储注意力系数的 TensorArray
        """用来将每一步输出处理成输入的函数
        """
        with tf.name_scope(name, "attention_decoder_fn_train", [time, cell_state, cell_input, cell_output, context_state]):
            # 第 0 个时间步之前的处理
            if cell_state is None:
                cell_state = encoder_state
                attention = init_attention(encoder_state)
                if output_alignments:
                    context_state = \
                        tf.TensorArray(dtype=tf.float32, tensor_array_name="alignments_ta",
                                       size=decoder_len, dynamic_size=True, infer_shape=False)
            # 之后的时间步之前的处理
            else:
                # 训练时，attention_construct_fn 返回 (拼接好的上下文，alignments) 的元组
                attention = attention_construct_fn(cell_output, keys, values)
                if output_alignments:
                    attention, alignments = attention
                    context_state = context_state.write(time-1, alignments)
                cell_output = attention
            next_input = tf.concat([cell_input, attention], 1)  # [batch_size, decoder_len, input_size]
            return (None, cell_state, next_input, cell_output, context_state)
    return decoder_fn

def attention_decoder_fn_inference(output_fn,  # project 函数
                                   encoder_state,  # 编码器输出状态 [layers, (c,h), batch, units]
                                   keys,  #
                                   values,  #
                                   attention_score_fn,
                                   attention_construct_fn,
                                   embeddings,  # 词嵌入
                                   start_of_sequence_id,  # GO_ID 2
                                   end_of_sequence_id,  # EOS_ID 3
                                   maximum_length,  # 解码最大允许的时间步
                                   num_symbol,  # num_symbols
                                   dtype=tf.int32,
                                   name=None):
    """推导时，用于 dynamic_rnn_decoder 的注意力 decoder 函数
    """
    with tf.name_scope(name, "attention_decoder_fn_inference",
                       [output_fn, encoder_state, keys, values, attention_score_fn, attention_construct_fn,
                        embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, num_symbol, dtype]):
        start_of_sequence_id = tf.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = tf.convert_to_tensor(end_of_sequence_id, dtype)

        maximum_length = tf.convert_to_tensor(maximum_length, dtype)
        num_symbol = tf.convert_to_tensor(num_symbol, dtype
                                          )
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value

        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = tf.shape(encoder_info)[0]

    def decoder_fn(time,  # 当前将要进行的时间步
                   cell_state,  # RNN 长时记忆，初始化时需要赋值
                   cell_input,  # 当前时间步的输入
                   cell_output,  # 上一步的输出
                   context_state):  # 存储注意力系数的 TensorArray
        """在 dynamic_rnn_decoder 中用于推导的 decoder 函数
        """
        with tf.name_scope(name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            # 在推导时，是没有输入的
            if cell_input is not None:
                raise ValueError("期待的 cell_input 是 None，但是 cell_input=%s" % cell_input)

            # 第 0 个时间步
            if cell_output is None:
                cell_state = encoder_state
                attention = init_attention(encoder_state)
                # 上个时间步的输出，第 0 步之前没有输出，所以初始化全 0
                cell_output = tf.zeros((num_symbol), dtype=tf.float32)  # [num_decoder_symbols] 全0
                # 这个时间步输入词汇的 id, 即 GO_ID
                next_input_id = tf.ones((batch_size), dtype=dtype) * start_of_sequence_id  # [batch_size] 全2
                # 将输入 id 转化成嵌入
                cell_input = tf.gather(embeddings, next_input_id)  # [batch_size, num_embed_units]
                # 是否已经结束生成
                done = tf.zeros((batch_size), dtype=tf.bool)  # [batch_size] 全 False
                # 存储输出 id 的 TensorArray
                context_state = \
                    tf.TensorArray(dtype=tf.int32, tensor_array_name="output_ids_ta",
                                   size=maximum_length, dynamic_size=True, infer_shape=False)

            # 之后的时间步
            else:
                attention = attention_construct_fn(cell_output, keys, values)
                cell_output = attention
                # [batch_size, num_symbols] 未 softmax 的预测
                cell_output = output_fn(cell_output)
                # [batch_size] 最大概率生成词的 id
                next_input_id = tf.cast(tf.argmax(cell_output, 1), dtype=dtype)
                # 保存输出的 id
                context_state = context_state.write(time-1, next_input_id)
                # 下个时间步细胞输入
                cell_input = tf.gather(embeddings, next_input_id)  # [batch_size, num_embed_units]
                # 是否已经结束生成
                done = tf.equal(next_input_id, end_of_sequence_id)
        # 下个时间步输入，加上 attention
        next_input = tf.concat([cell_input, attention], 1)

        # 如果 time > maximum_length 则返回全为 True 的向量，否则返回 done
        done = tf.cond(tf.greater(time, maximum_length),
                lambda: tf.ones((batch_size), dtype=tf.bool),
                lambda: done)
        return (done, cell_state, next_input, cell_output, context_state)
    return decoder_fn


def prepare_attention(encoder_output,  # 编码器输出 [batch_size, encoder_len, num_units]
                      num_units,
                      attention_option="bahdanau",
                      output_alignments=False,
                      reuse=False):
    # 根据编码器的输出，构造注意力的 keys 和 values
    with tf.variable_scope("attention_keys", reuse=reuse) as scope:
        attention_keys = layers.linear(encoder_output, num_units, biases_initializer=None, scope=scope)
        attention_values = encoder_output

    attention_score_fn = create_attention_score_fn(num_units, attention_option=attention_option,
                                                   output_alignments=output_alignments, reuse=reuse)
    attention_construct_fn = create_attention_construct_fn(num_units, attention_score_fn, reuse=reuse)

    return (attention_keys, attention_values, attention_score_fn, attention_construct_fn)


def create_attention_score_fn(num_units,
                              attention_option="bahdanau",
                              output_alignments=False,
                              dtype=tf.float32,
                              reuse=False):
    with tf.variable_scope("attention_score", reuse=reuse):
        if attention_option == "bahdanau":
            query_w = tf.get_variable(name="attnW", shape=(num_units, num_units), dtype=dtype)
            score_v = tf.get_variable(name="attnV", shape=(num_units), dtype=dtype)

        # 通过计算 query 和 keys 的相关系数，然后作用在 value 上获得上下文
        def attention_score_fn(query,  # 解码器输出 [batch_size, num_units]
                               keys,  # [batch_size, encoder_len, num_units]
                               values):  # [batch_size, encoder_len, num_units]
                # 计算注意力分数
                if attention_option == "bahdanau":
                    query = tf.matmul(query, query_w)  # [batch_size, num_units]
                    query = tf.reshape(query, [-1, 1, num_units])  # [batch_size, 1, num_units]
                    scores = tf.reduce_sum(score_v * tf.tanh(keys + query), [2])  # [batch_size, encoder_len]
                elif attention_option == "luong":
                    query = tf.reshape(query, [-1, 1, num_units])  # [batch_size, 1, num_units]
                    scores = tf.reduce_sum(keys * query, [2])  # [batch_size, encoder_len]
                else:
                    raise ValueError("未知的注意力机制 %s!" % attention_option)
                # 计算注意力权重
                alignments = tf.nn.softmax(scores)  # [batch_size, encoder_len]
                new_alignments = tf.expand_dims(alignments, 2)  # [batch_size, encoder_len, 1]
                # 计算上下文
                context = tf.reduce_sum(new_alignments * values, [1])  # batch_size * num_units
                context.set_shape([None, num_units])
                if output_alignments:
                    return (context, alignments)
                else:
                    return context
        return attention_score_fn

def create_attention_construct_fn(num_units,
                                  attention_score_fn,
                                  reuse=False):
    with tf.variable_scope("attention_construct", reuse=reuse) as scope:
        # 拼接计算完的上下文
        def attention_construct_fn(query,  # 解码器输出 [batch_size, num_units]
                                   keys,  # [batch_size, encoder_len, num_units]
                                   values):  # [batch_size, encoder_len, num_units]
            alignments = None
            context = attention_score_fn(query, keys, values)
            if type(context) is tuple:
                context, alignments = context
            concat_input = tf.concat([query, context], axis=1)
            attention = layers.linear(concat_input, num_units, biases_initializer=None,
                                      scope=scope)  # [batch_size, num_units]
            if alignments is None:
                return attention  # [batch_size, num_units]
            else:
                return attention, alignments

        return attention_construct_fn

# 初始化注意力
def init_attention(encoder_state):  # 编码器状态 [layers, (c,h), batch_size, num_units]
    # top_state: [(c,h), batch_size, num_units]
    if isinstance(encoder_state, tuple):  # 多层的编码器
        top_state = encoder_state[-1]
    else:  # 单层的编码器
        top_state = encoder_state
    # attn: [batch_size, num_units]
    if isinstance(top_state, rnn_cell_impl.LSTMStateTuple):  # LSTM
        attn = tf.zeros_like(top_state.h)
    else:  # GRU
        attn = tf.zeros_like(top_state)
    return attn  # [batch_size, num_units]





