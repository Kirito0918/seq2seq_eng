# 动态 RNN 解码器
import tensorflow as tf

def dynamic_rnn_decoder(cell,  # 多层的 RNNCell
                        decoder_fn,  # 对每个时间步输出进行处理成输入的函数
                        inputs=None,  # 训练时，传入该参数，为 response 的嵌入向量 [batch_size, decoder_len, embedding_size]
                        sequence_length=None,  # 训练时，传入该参数，为 response 的长度向量
                        parallel_iterations=None,  # 平行运行中的迭代数量
                        swap_memory=False,
                        time_major=False,  # 输入的数据集是否是 time-major 的
                        scope=None,  # 变量空间
                        name=None):  # 命名空间
    """ seq2seq 模型的 RNN 动态解码器.
    """
    with tf.name_scope(name, "dynamic_rnn_decoder",
                       [cell, decoder_fn, inputs, sequence_length, parallel_iterations, swap_memory, time_major, scope]):
        if inputs is not None:  # 训练时，解码器有输入
            inputs = tf.convert_to_tensor(inputs)
            if inputs.get_shape().ndims is not None and (
                    inputs.get_shape().ndims < 2):
                raise ValueError("输入至少有2维")

            if not time_major:
                # [batch, seq, features] -> [seq, batch, features]
                inputs = tf.transpose(inputs, perm=[1, 0, 2])  # [decoder_len, batch_size, embedding_size]

            dtype = inputs.dtype
            input_depth = int(inputs.get_shape()[2])  # embedding_size 输入维度
            batch_depth = inputs.get_shape()[1].value  # batch_size 样本个数
            max_time = inputs.get_shape()[0].value  # decoder_len 解码器最大时间步
            if max_time is None:
                max_time = tf.shape(inputs)[0]

            # 将解码器的输入设置成一个 tensor 数组
            # 数组长度为 decoder_len，数组的每个元素是个 [batch_size, embedding_size] 的张量
            inputs_ta = tf.TensorArray(dtype, size=max_time)
            inputs_ta = inputs_ta.unstack(inputs)

###############################################动态rnn需要复写的函数
        def loop_fn(time,  # 第 time 个时间步之前的处理，起始为 0
                    cell_output,  # 上一个时间步的输出
                    cell_state,  # RNNCells 的长时记忆
                    loop_state):  # 保存了上个时间步执行后是否已经结束，如果输出 alignments，还保存了存有 alignments 的 TensorArray
            """loop_fn 是一个函数，这个函数在 rnn 的相邻时间步之间被调用。
            """
            # 第 0 次解码之前调用
            if cell_state is None:
                if cell_output is not None:
                    raise ValueError("当 cell_state is None 时，cell_output 应当为 None，但是 cell_output = %s" % cell_output)
                if loop_state is not None:
                    raise ValueError("当 cell_state is None 时，loop_state 应当为 None，但是 loop_state = %s" % loop_state)
                context_state = None

            # 后续解码之前的调用
            else:
                if isinstance(loop_state, tuple):  # 如果循环的状态是个元组，即包含 done 和上下文信息
                    (done, context_state) = loop_state
                else:  # 如果循环状态不是个元组，即只包含 done 的信息
                    done = loop_state
                    context_state = None

            # 训练时
            if inputs is not None:
                # 第 0 个时间步之前的处理
                if cell_state is None:
                    next_cell_input = inputs_ta.read(0)  # 读取输入的第一列，即 GO_ID

                # 后续时间步之前的处理
                else:
                    if batch_depth is not None:
                        batch_size = batch_depth
                    else:
                        batch_size = tf.shape(done)[0]  # done 是对循环是否结束的标注，
                    # 如果 time == max_time, 则 next_cell_input = batch_size * embedding_size 的全 1 矩阵
                    # 否则，next_cell_input 从数据中读取这个时间步的数据
                    next_cell_input = tf.cond(
                            tf.equal(time, max_time),
                            lambda: tf.zeros([batch_size, input_depth], dtype=dtype),
                            lambda: inputs_ta.read(time))

                # emit_output = attention
                (next_done, next_cell_state, next_cell_input, emit_output, next_context_state) = \
                    decoder_fn(time, cell_state, next_cell_input, cell_output, context_state)

            # 推导时
            else:
                (next_done, next_cell_state, next_cell_input, emit_output, next_context_state) = \
                    decoder_fn(time, cell_state, None, cell_output, context_state)

            # 检查这个时间步是否已经结束
            if next_done is None:  # 当训练时，next_done 返回的是 None
                next_done = time >= sequence_length  # [batch_size]

            if next_context_state is None:
                next_loop_state = next_done
            else:
                next_loop_state = (next_done, next_context_state)

            return (next_done, next_cell_input, next_cell_state,
                            emit_output, next_loop_state)
########################################################################################################################
        # 运行 raw_rnn
        outputs_ta, final_state, final_loop_state = \
            tf.nn.raw_rnn(cell, loop_fn, parallel_iterations=parallel_iterations, swap_memory=swap_memory, scope=scope)
        outputs = outputs_ta.stack()

        # 如果要输出 alignments，就获取 final_context_state
        if isinstance(final_loop_state, tuple):
            final_context_state = final_loop_state[1]
        else:
            final_context_state = None

        if not time_major:
            # [seq, batch, features] -> [batch, seq, features]
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        return outputs, final_state, final_context_state