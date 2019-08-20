
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs

__all__ = ["dynamic_rnn_decoder"]

def dynamic_rnn_decoder(cell,  # 多层的 RNNCell
                        decoder_fn,  # 对每个时间步输出进行处理成输入的函数
                        inputs=None,  # 训练时，传入该参数，为 response 的嵌入向量 [batch_size, decoder_len, 600（300为词嵌入，100*3为3个实体嵌入）]
                        sequence_length=None,  # 训练时，传入该参数，为 response 的长度向量
                        parallel_iterations=None,  # 没用到这个参数
                        swap_memory=False,  # 没用到这个参数
                        time_major=False,  # 表示输入的数据集是否是 time-major 的，实验中为 False
                        scope=None,  # ="decoder_rnn"
                        name=None):  # 没用到这个参数
    with ops.name_scope(name, "dynamic_rnn_decoder",
                                            [cell, decoder_fn, inputs, sequence_length,
                                             parallel_iterations, swap_memory, time_major, scope]):
        if inputs is not None:
            # 将输入转化成张量
            inputs = ops.convert_to_tensor(inputs)
            # 测试输入的维度,不能小于 2
            if inputs.get_shape().ndims is not None and (
                    inputs.get_shape().ndims < 2):
                raise ValueError("Inputs must have at least two dimensions")

            # 如果不是 time_major，就要做一个转置
            if not time_major:
                # [batch, seq, features] -> [seq, batch, features]
                inputs = array_ops.transpose(inputs, perm=[1, 0, 2])  # decoder_len * batch_size * 600

            dtype = inputs.dtype
            input_depth = int(inputs.get_shape()[2])  # 600
            batch_depth = inputs.get_shape()[1].value  # batch_size
            max_time = inputs.get_shape()[0].value  # decoder_len
            if max_time is None:
                max_time = array_ops.shape(inputs)[0]

            # 将解码器的输入设置成一个 tensor 数组
            # 数组长度为 decoder_len，数组的每个元素是个 batch_size * 600 的张量
            inputs_ta = tensor_array_ops.TensorArray(dtype, size=max_time)
            inputs_ta = inputs_ta.unstack(inputs)

        def loop_fn(time, cell_output, cell_state, loop_state):
            # 解码之前第一次调用
            if cell_state is None:
                # cell_state is None 时，cell_output 应该为 None
                if cell_output is not None:
                    raise ValueError("Expected cell_output to be None when cell_state "
                                                     "is None, but saw: %s" % cell_output)
                # cell_state is None 时，loop_state 应该为 None
                if loop_state is not None:
                    raise ValueError("Expected loop_state to be None when cell_state "
                                                     "is None, but saw: %s" % loop_state)
                context_state = None

            # 后续的调用
            else:
                if isinstance(loop_state, tuple):
                    (done, context_state) = loop_state
                else:
                    done = loop_state
                    context_state = None

            # 训练
            # 训练时 input is not None
            # 获得 next_cell_input
            if inputs is not None:
                # 第一个时间步之前的处理
                if cell_state is None:
                    next_cell_input = inputs_ta.read(0)  # 其实第一列都是 GO_ID

                # 之后的 cell 之间的处理
                else:

                    if batch_depth is not None:
                        batch_size = batch_depth
                    else:
                        batch_size = array_ops.shape(done)[0]  # done 是对循环是否结束的标注，

                    # 如果 time == max_time, 则 next_cell_input = batch_size * 600 的全 1 矩阵
                    # 否则，next_cell_input 从数据中读下一时间步的数据
                    next_cell_input = control_flow_ops.cond(
                            math_ops.equal(time, max_time),
                            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtype),
                            lambda: inputs_ta.read(time))

                # emit_output = attention
                (next_done, next_cell_state, next_cell_input, emit_output,
                 next_context_state) = decoder_fn(time, cell_state, next_cell_input, cell_output, context_state)
            # 推导
            else:
                # next_cell_input 通过 decoder_fn 获得
                (next_done, next_cell_state, next_cell_input, emit_output,
                 next_context_state) = decoder_fn(time, cell_state, None, cell_output,
                                                                                    context_state)
            # 检查是否已经结束
            if next_done is None:  # 当训练时，next_done 返回的是 None
                next_done = time >= sequence_length  # 当 time >= sequence_length 时，next_done = True

            # 构建 next_loop_state
            if next_context_state is None:  # 如果不输出 alignments
                next_loop_state = next_done
            else:
                next_loop_state = (next_done, next_context_state)

            return (next_done, next_cell_input, next_cell_state,
                            emit_output, next_loop_state)

        # Run raw_rnn function
        outputs_ta, final_state, final_loop_state = rnn.raw_rnn(
                cell, loop_fn, parallel_iterations=parallel_iterations,
                swap_memory=swap_memory, scope=scope)
        outputs = outputs_ta.stack()

        # 如果要输出 alignments，就获取 final_context_state
        if isinstance(final_loop_state, tuple):
            final_context_state = final_loop_state[1]
        else:
            final_context_state = None

        # 如果不是 time_major，就转置回去
        if not time_major:
            # [seq, batch, features] -> [batch, seq, features]
            outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
        return outputs, final_state, final_context_state
