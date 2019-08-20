from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def attention_decoder_fn_train(encoder_state,
                             attention_keys,
                             attention_values,
                             attention_score_fn,
                             attention_construct_fn,
                             output_alignments=False,
                             max_length=None,
                             name=None):
    # 当 name is None 时，name_scope 的默认名为 attention_decoder_fn_train，并把列表中的操作加入到该 name_scope
    with ops.name_scope(name, "attention_decoder_fn_train", [
            encoder_state, attention_keys, attention_values, attention_score_fn,
            attention_construct_fn
    ]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(name, "attention_decoder_fn_train", [time, cell_state, cell_input, cell_output, context_state]):
            # 第 0 个时间步
            if cell_state is None:
                cell_state = encoder_state  # 用 encoder_state 去初始化 cell_state，这个操作很强

                # 采用初始化的 attention
                attention = _init_attention(encoder_state)
                # 如果要输出 alignments，则声明一个叫 context_state 的 TensorArray 用来记录
                if output_alignments:
                    context_state = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name="alignments_ta", size=max_length, dynamic_size=True, infer_shape=False)

            # 之后的时间步
            else:
                # 训练时，attention_construct_fn 返回 (拼接好的上下文，alignments) 的元组
                attention = attention_construct_fn(cell_output, attention_keys, attention_values)
                if output_alignments:
                    attention, alignments = attention
                    context_state = context_state.write(time-1, alignments)  # 记录一下 alignments
                cell_output = attention

            # 拼接 cell_input 和 attention，成为下一步的输入
            next_input = array_ops.concat([cell_input, attention], 1)  # [batch_size, decoder_len, 600+num_units]

            return (None, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def attention_decoder_fn_inference(output_fn,  #
                                 encoder_state,  # 编码器状态
                                 attention_keys,  #
                                 attention_values,  #
                                 attention_score_fn,  # 计算上下文的函数
                                 attention_construct_fn,  # 计算 attention 的函数
                                 embeddings,  # 词嵌入
                                 start_of_sequence_id,  # GO_ID 2
                                 end_of_sequence_id,  # EOS_ID 3
                                 maximum_length,  # 60
                                 num_decoder_symbols,  # num_symbols
                                 dtype=dtypes.int32,
                                 selector_fn=None,  # 选择知识图中实体词的概率
                                 imem=None,  # ([batch_size,triple_num*triple_len,num_embed_units],[encoder_batch_size, triple_num*triple_len, 3*num_trans_units]) 实体嵌入和三元组嵌入的元组
                                 name=None):
    with ops.name_scope(name, "attention_decoder_fn_inference", [
            output_fn, encoder_state, attention_keys, attention_values,
            attention_score_fn, attention_construct_fn, embeddings, imem,
            start_of_sequence_id, end_of_sequence_id, maximum_length,
            num_decoder_symbols, dtype
    ]):
        # 将一些数值转化成张量
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)

        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value

        # 如果 output_fn 为 None 则做一个恒等变换
        if output_fn is None:
            output_fn = lambda x: x

        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(
                name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            # 在推导时，是没有输入的
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                                 cell_input)
            # time = 0
            if cell_output is None:
                next_input_id = array_ops.ones(
                        [batch_size,], dtype=dtype) * (start_of_sequence_id)  # batch_size 全2

                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)  # batch_size 全 False
                cell_state = encoder_state  # 用编码器状态初始化 RNNCell state
                cell_output = array_ops.zeros(
                        [num_decoder_symbols], dtype=dtypes.float32)  # [num_decoder_symbols]
                # 对于 next_input_id 中的每个 value，从 embeddings 中找 embeddings[value] 回填到 next_input_id 的位置产生一个 list
                word_input = array_ops.gather(embeddings, next_input_id)  # batch_size * num_embed_units

                naf_triple_id = array_ops.zeros([batch_size, 2], dtype=dtype)  # [batch_size, 2] 的全 0 矩阵
                # imem[1]: [encoder_batch_size, triple_num*triple_len, 3*num_trans_units] 三元组嵌入
                triple_input = array_ops.gather_nd(imem[1], naf_triple_id)  # [batch_size, 3*num_trans_units]
                cell_input = array_ops.concat([word_input, triple_input], axis=1)  # [batch_size, num_embed_units+3*num_trans_units]
                # 初始化注意力
                attention = _init_attention(encoder_state)
                if imem is not None:  # 如果传入了实体嵌入和词嵌入
                    context_state = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="output_ids_ta", size=maximum_length, dynamic_size=True, infer_shape=False)
            # time >= 1
            else:
                # 构建注意力
                attention = attention_construct_fn(cell_output, attention_keys, attention_values)
                if type(attention) is tuple:  # 输出 alignment 的情况
                    attention, alignment = attention
                    cell_output = attention
                    alignment = tf.reshape(alignment, [batch_size, -1])  # [batch_size, triple_num*triple_len]
                    selector = selector_fn(cell_output)  # 选择实体词的概率选择器
                    logit = output_fn(cell_output)  # [batch_size, num_decoder_symbols] 未 softmax 的预测
                    word_prob = nn_ops.softmax(logit) * (1 - selector)  # [batch_size, num_decoder_symbols] 选择生成词概率
                    entity_prob = alignment * selector  # 选择实体词的概率

                    # 这步操作对生成词和实体词的使用概率进行的一个比较，选择其中概率最大的，从而形成了一个 mask
                    # 1、tf.reduce_max(word_prob, 1): [batch_size] 生成词最大的概率
                    # 2、tf.reduce_max(entity_prob, 1): [batch_size] 实体词最大的概率
                    # 3、greater: [batch_size] 生成词的概率是否大于实体词概率
                    # 4、cast: [batch_size] 将 bool 值转化成浮点
                    # 5、reshape(cast): [batch_size， 1] 用生成词则为 1，否则则为0
                    mask = array_ops.reshape(math_ops.cast(math_ops.greater(tf.reduce_max(word_prob, 1), tf.reduce_max(entity_prob, 1)), dtype=dtypes.float32), [-1,1])

                    # 1、cast(math_ops.argmax(word_prob, 1): [batch_size] 生成词中最大概率的下标
                    # 2、gather: [batch_size， num_embed_units]: 采用的生成词
                    # 3、mask * gather: [batch_size, num_embed_units] 实际采用的生成词
                    # 4、reshape(range(batch_size)): [batch_size, 1]
                    # 5、reshape(cast(argmax(entity_prob, 1))): [batch_size, 1] 实体词中最大概率的下标
                    # 6、cast: [batch_size, 2] 4、5 两步的结果在第 1 维度上拼接
                    # 7、imem[0]:[batch_size, triple_num*triple_len, num_embed_units]
                    # 8、gather_nd: [batch_size, num_embed_units] 采用的实体词
                    # 9、(1-mask) * gather_nd: 实际采用的生成词
                    # 10、mask * gather + (1-mask) * gather_nd: [batch_size, num_embed_units] 该时间步的实际输出
                    word_input = mask * array_ops.gather(embeddings, math_ops.cast(math_ops.argmax(word_prob, 1), dtype=dtype)) \
                                 + (1 - mask) * array_ops.gather_nd(imem[0], array_ops.concat([array_ops.reshape(math_ops.range(batch_size, dtype=dtype), [-1,1]), array_ops.reshape(math_ops.cast(math_ops.argmax(entity_prob, 1), dtype=dtype), [-1,1])], axis=1))

                    # 1、reshape(range(batch_size)): [batch_size, 1]
                    # 2、cast(1-mask): [batch_size, 1] 选择实体词的 mask
                    # 3、reshape(argmax(alignment, 1)): [batch_size, 1] 选择实体词的下标
                    # 4、cast(1-mask) * reshape(argmax(alignment, 1)): [batch_size, 1] 选择了实体词，则为实体词下标，否则则为0
                    # 5、concat: [batch_size, 2] 第二个维度的第一个元素为 batch，第二个元素为 indice
                    indices = array_ops.concat([array_ops.reshape(math_ops.range(batch_size, dtype=dtype), [-1,1]), math_ops.cast(1-mask, dtype=dtype) * tf.reshape(math_ops.cast(math_ops.argmax(alignment, 1), dtype=dtype), [-1, 1])], axis=1)

                    # imem[1]: [encoder_batch_size, triple_num*triple_len, 3*num_trans_units] 三元组嵌入
                    # 使用的三元组嵌入
                    triple_input = array_ops.gather_nd(imem[1], indices)  # [batch_size, 3*num_trans_units]
                    # 下个时间步细胞输入
                    cell_input = array_ops.concat([word_input, triple_input], axis=1)  # [batch_size, num_embed_units+3*num_trans_units]
                    mask = array_ops.reshape(math_ops.cast(mask, dtype=dtype), [-1])  # [batch_size] 选择生成词的 mask

                    # argmax(word_prob, 1): [batch_size] 生成词下标
                    # mask - 1: [batch_size] 如果取生成词则为 0，如果取实体词则为 -1
                    # argmax(entity_prob, 1): [batch_size] 实体词下标
                    # input_id: [batch_size] 如果为生成词则 id 为正，如果为实体词则 id 为负
                    input_id = mask * math_ops.cast(math_ops.argmax(word_prob, 1), dtype=dtype) + (mask - 1) * math_ops.cast(math_ops.argmax(entity_prob, 1), dtype=dtype)

                    # 把 input_id 写入 TensorArray
                    context_state = context_state.write(time-1, input_id)
                    # 判断句子是否已经结束
                    done = array_ops.reshape(math_ops.equal(input_id, end_of_sequence_id), [-1])
                    cell_output = logit  # [batch_size, num_decoder_symbols] 未 softmax 的预测
                else:  # 不输出 alignments 的情况
                    cell_output = attention

                    # argmax decoder
                    cell_output = output_fn(cell_output)  # [batch_size, num_decoder_symbols] 未 softmax 的预测
                    # [batch_size] 最大概率生成词的下标
                    next_input_id = math_ops.cast(
                            math_ops.argmax(cell_output, 1), dtype=dtype)
                    # 判断句子是否已经结束
                    done = math_ops.equal(next_input_id, end_of_sequence_id)
                    # 下个时间步细胞输入
                    cell_input = array_ops.gather(embeddings, next_input_id)  # [batch_size, num_embed_units]

            # 下个时间步输入，加上 attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # 如果 time > maximum_length 则返回全为 True 的向量，否则返回 done
            done = control_flow_ops.cond(
                    math_ops.greater(time, maximum_length),
                    lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                    lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn

def prepare_attention(attention_states,  # 编码器输出: 每个 batch，每个时间步的输出
                          attention_option,  # 'bahdanau'
                          num_units,
                          imem=None,  # (graph_embed, triples_embedding) graph_embed: 图的100维向量表示  [batch_size, triple_num, 100] triples_embedding: [encoder_batch_size, triple_num, triple_len, 3*num_trans_units]
                          output_alignments=False,  # 训练时为 True
                          reuse=False):

    with variable_scope.variable_scope("attention_keys", reuse=reuse) as scope:
        # 初始化上下文的 attention_keys/attention_values: [batch_size, encoder_len, num_units]
        attention_keys = layers.linear(
            attention_states, num_units, biases_initializer=None, scope=scope)
        attention_values = attention_states

    if imem is not None:
        if type(imem) is tuple:
            # imem[0]: [batch_size, triple_num, 100] 图的 100 维向量表示
            # 初始化图的 key/value: [batch_size, triple_num, num_units]
            with variable_scope.variable_scope("imem_graph", reuse=reuse) as scope:
                attention_keys2, attention_states2 = array_ops.split(layers.linear(
                    imem[0], num_units*2, biases_initializer=None, scope=scope), [num_units, num_units], axis=2)

            # imem[1]: [encoder_batch_size, triple_num, triple_len, 3 * num_trans_units] 三元组嵌入
            # 初始化三元组的 key/value: [encoder_batch_size, triple_num, triple_len, num_units]
            with variable_scope.variable_scope("imem_triple", reuse=reuse) as scope:
                attention_keys3, attention_states3 = array_ops.split(layers.linear(
                    imem[1], num_units*2, biases_initializer=None, scope=scope), [num_units, num_units], axis=3)

            attention_keys = (attention_keys, attention_keys2, attention_keys3)
            attention_values = (attention_states, attention_states2, attention_states3)
        else:
            with variable_scope.variable_scope("imem", reuse=reuse) as scope:
                attention_keys2, attention_states2 = array_ops.split(layers.linear(
                    imem, num_units*2, biases_initializer=None, scope=scope), [num_units, num_units], axis=2)
                attention_keys = (attention_keys, attention_keys2)
                attention_values = (attention_states, attention_states2)

    if imem is None:  # 如果没有图向量和三元组嵌入
        attention_score_fn = _create_attention_score_fn("attention_score", num_units,
                                                            attention_option, reuse)
    else:  # 训练时走的这条分支
        attention_score_fn = (_create_attention_score_fn("attention_score", num_units,
                                                            attention_option, reuse),  # 这个函数用来计算上下文，不输出 alignments
                            _create_attention_score_fn("imem_score", num_units,
                                                            "luong", reuse, output_alignments=output_alignments))  # 这个函数用来计算图上下文和三元组上下文

    # 这个函数用来计算拼接完的上下文
    attention_construct_fn = _create_attention_construct_fn("attention_construct",
                                  num_units,
                                  attention_score_fn,
                                  reuse)

    return (attention_keys, attention_values, attention_score_fn,
                    attention_construct_fn)

# def _init_attention(encoder_state):
#     # Multi- vs single-layer
#     # TODO(thangluong): is this the best way to check?
#     if isinstance(encoder_state, tuple):  # 多层的编码器，选择顶层的 StateTuple
#         top_state = encoder_state[-1]
#     else:  # 单层的解码器，就不用选择了
#         top_state = encoder_state
#
#     # LSTM vs GRU
#     if isinstance(top_state, rnn_cell_impl.LSTMStateTuple):  # LSTM
#         attn = array_ops.zeros_like(top_state.h)
#     else:  # GRU
#         attn = array_ops.zeros_like(top_state)
#
#     return attn


def _create_attention_construct_fn(name, num_units, attention_score_fn, reuse):
    with variable_scope.variable_scope(name, reuse=reuse) as scope:

        def construct_fn(attention_query, attention_keys, attention_values):
            alignments = None
            # 如果有图向量和三元组嵌入
            if type(attention_score_fn) is tuple:
                # 用 bahdanau 计算带注意力的上下文
                context0 = attention_score_fn[0](attention_query, attention_keys[0],
                                                                         attention_values[0])
                # 训练没走这个分支
                if len(attention_keys) == 2:
                    context1 = attention_score_fn[1](attention_query, attention_keys[1],
                                                                             attention_values[1])
                # 如果有图向量和三元组嵌入
                elif len(attention_keys) == 3:
                    context1 = attention_score_fn[1](attention_query, attention_keys[1:],
                            attention_values[1:])

                if type(context1) is tuple:
                    if len(context1) == 2:
                        context1, alignments = context1
                        concat_input = array_ops.concat([attention_query, context0, context1], 1)
                    elif len(context1) == 3:  # 训练走的这个分支
                        context1, context2, alignments = context1
                        concat_input = array_ops.concat([attention_query, context0, context1, context2], 1)  # [batch_size, num_units*4] 解码器输出，上下文，图上下文，三元组上下文拼接
                else:
                    concat_input = array_ops.concat([attention_query, context0, context1], 1)
            else:
                context = attention_score_fn(attention_query, attention_keys,
                                                                         attention_values)
                concat_input = array_ops.concat([attention_query, context], 1)
            # 给拼接完的分布做一个线性变化，将最后一个维度转为 num_units
            attention = layers.linear(
                    concat_input, num_units, biases_initializer=None, scope=scope)  # [batch_size, num_units]
            if alignments is None:
                return attention
            else:
                return attention, alignments

        return construct_fn


@function.Defun(func_name="attn_add_fun", noinline=True)
def _attn_add_fun(v, keys, query):
    return math_ops.reduce_sum(v * math_ops.tanh(keys + query), [2])
@function.Defun(func_name="attn_mul_fun", noinline=True)
def _attn_mul_fun(keys, query):
    return math_ops.reduce_sum(keys * query, [2])

def _create_attention_score_fn(name,
                                   num_units,
                                   attention_option,
                                   reuse,
                                   output_alignments=False,  # 是否将 alignment 输出
                                   dtype=dtypes.float32):
    with variable_scope.variable_scope(name, reuse=reuse):
        if attention_option == "bahdanau":
            query_w = variable_scope.get_variable(
                    "attnW", [num_units, num_units], dtype=dtype)
            score_v = variable_scope.get_variable("attnV", [num_units], dtype=dtype)

        def attention_score_fn(query, keys, values):
            triple_keys, triple_values = None, None
            if type(keys) is tuple:
                keys, triple_keys = keys
                values, triple_values = values

            if attention_option == "bahdanau":
                query = math_ops.matmul(query, query_w)
                query = array_ops.reshape(query, [-1, 1, num_units])
                scores = _attn_add_fun(score_v, keys, query)
            elif attention_option == "luong":
                query = array_ops.reshape(query, [-1, 1, num_units])
                scores = _attn_mul_fun(keys, query)
            else:
                raise ValueError("Unknown attention option %s!" % attention_option)

            alignments = nn_ops.softmax(scores)
            new_alignments = array_ops.expand_dims(alignments, 2)
            context_vector = math_ops.reduce_sum(new_alignments * values, [1])  # batch_size * num_units
            context_vector.set_shape([None, num_units])

            # 计算三元组上下文
            if triple_values is not None:
                # triple_keys: [batch_size, triple_num, triple_len, num_units]
                triple_scores = math_ops.reduce_sum(triple_keys * array_ops.reshape(query, [-1, 1, 1, num_units]), [3])  # [batch_size, triple_num, triple_len]
                triple_alignments = nn_ops.softmax(triple_scores)  # [batch_size, triple_num, triple_len]
                # 计算每个三元组的上下文
                context_triples = math_ops.reduce_sum(array_ops.expand_dims(triple_alignments, 3) * triple_values, [2])  # [batch_size, triple_num, num_units]
                # 加上对图注意力系数的三元组上下文
                context_graph_triples = math_ops.reduce_sum(new_alignments * context_triples, [1])
                context_graph_triples.set_shape([None, num_units])

                # 对图的注意力系数 * 对三元组的注意力系数
                final_alignments = new_alignments * triple_alignments  # [batch_size, triple_num, triple_len]
                return context_vector, context_graph_triples, final_alignments
            else:
                if output_alignments:
                    return context_vector, alignments  # 对时间步的注意力系数/对图的注意力系数
                else:
                    return context_vector  # 上下文/图上下文

        return attention_score_fn
