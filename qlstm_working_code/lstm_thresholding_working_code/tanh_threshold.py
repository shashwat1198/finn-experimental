#This file contains 'Tanh + QuantLinear' onnx graph for comparison against the multithresholding node.

import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from qonnx.util.basic import qonnx_make_model
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import version_converter, helper

# Input definition
inp_X = make_tensor_value_info(
"X",onnx.TensorProto.FLOAT, [256,1]
)

#Output definition
out_tanh = make_tensor_value_info(
"t_t", onnx.TensorProto.FLOAT, [256,1]
)

quant_out_tanh = make_tensor_value_info(
"quant_out_tanh", onnx.TensorProto.FLOAT, [256,1]
)

quant_out_ql = make_tensor_value_info(
"quant_out_ql", onnx.TensorProto.INT8, [256,1]
) # Output of QuantLinear layer. Therefore INT8 datatype

#Defining the individual nodes of the multi-threshold graph we want to create.
# --------------------------------------------

tanh_s = make_node(
    "Tanh", inputs=["X"], outputs=["t_t"],name="tanh_s"
)
ql_act = make_node(
    "QuantizeLinear", inputs=["t_t","scale_tanh","zero_point_tanh"], outputs=["quant_out_ql"], name="ql_act"
    )

sigmoid_mt_graph = onnx.helper.make_graph(
    nodes=[ 
           tanh_s,
           ql_act
          ],
    name = "tanh-mt-graph",
    inputs=[inp_X],#The order of the inputs reversed here in order to match the order of inputs of the defined scan node.
    outputs = [out_tanh,quant_out_ql],
    value_info=[
            make_tensor_value_info("ql_s_t", onnx.TensorProto.INT8, [20,1]),
            make_tensor_value_info("clp_s_t", onnx.TensorProto.INT8, [20,1]),
            make_tensor_value_info("dql_s_t",onnx.TensorProto.FLOAT, [20,1])
        ],
    initializer=[
                 make_tensor('scale_tanh',onnx.TensorProto.FLOAT,[],[0.0078125]),#128
                 make_tensor('zero_point_tanh',onnx.TensorProto.INT8,[],[0]),
                 make_tensor('min',onnx.TensorProto.INT8, [],[-128]),
                 make_tensor('max',onnx.TensorProto.INT8, [], [127]),
                 make_tensor('bitwidth',onnx.TensorProto.INT32, [], [8])
                ]
)

onnx_model = qonnx_make_model(sigmoid_mt_graph, producer_name="LSTM_eq")
# Have to convert the opset version of the graph here because the clip operator in the previous version did not allow for INT8 inputs.
# It only allowed for FLOAT inputs.
#onnx_model_14 = version_converter.convert_version(onnx_model, 14)
#onnx.save(onnx_model_14,'./sigmoid_threshold.onnx')

acivation_bit_width = 8
#Testing against the complete INT8 input domain range [-128,127]
in_X = np.arange(start= - 2 ** (acivation_bit_width - 1), stop= 2 ** (acivation_bit_width - 1), dtype=np.float32)

#Testing for the INT8 input domain of [-128,127]
in_X = in_X.reshape([256,1])
input_dict = {}
input_dict["X"] = in_X

sess = rt.InferenceSession(onnx_model.SerializeToString())
output = sess.run(None, input_dict)
print('Tanh output')
tanh_out = np.array(output[0])
print(tanh_out) 
print('QuantLinear output')
ql_out = np.array(output[1])
print(ql_out)