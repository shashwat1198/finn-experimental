import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from qonnx.util.basic import qonnx_make_model
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import version_converter, helper

ql_w = make_node("QuantizeLinear", inputs=["W_s","scale_all","zero_point_all"], outputs=["ql_ws"], name="ql_w")
clp_w = make_node("Clip", inputs=["ql_ws","min","max"], outputs=["clp_ws"], name="clp_ws")
dql_w = make_node("DequantizeLinear", inputs=["clp_ws","scale_all","zero_point_all"], outputs=["dql_ws"], name="dql_w")

ql_u = make_node("QuantizeLinear", inputs=["U_s","scale_all","zero_point_all"], outputs=["ql_us"], name="ql_u")
clp_u = make_node("Clip", inputs=["ql_us","min","max"], outputs=["clp_us"], name="clp_u")
dql_u = make_node("DequantizeLinear", inputs=["clp_us","scale_all","zero_point_all"], outputs=["dql_us"], name="dql_u")

# Defining the inputs and outputs of the graph we need to create.
# Input definition
inp_X = make_tensor_value_info(
"X",onnx.TensorProto.FLOAT, [10,1]
)

inp_h_t_1 = make_tensor_value_info(
"h_t-1",onnx.TensorProto.FLOAT, [20,1]
)

#Output definition

out_state = make_tensor_value_info(
"s_t", onnx.TensorProto.FLOAT, [20,1]
)

scan_out = make_tensor_value_info(
"scan_out", onnx.TensorProto.FLOAT, [20,1]
)

ql_s_t_out = make_tensor_value_info(
"ql_s_t_out", onnx.TensorProto.INT8, [20,1]
)

dql_s_t_add_out = make_tensor_value_info(
"dql_s_t_add_out", onnx.TensorProto.FLOAT, [20,1]
)


#Defining the individual nodes of the graph we want to create.
# --------------------------------------------
mul_node1 = make_node(
    "MatMul", inputs=["dql_ws","X"], outputs=["out_m1"], name="mul_node1" #For this graph to pass through FINN..all the initializers should be on the right side of the function call
) #Can't reverse inputs ONNX will not generate graphs otherwise.
mul_node2 = make_node(
    "MatMul", inputs=["dql_us","h_t-1"], outputs=["out_m2"],name="mul_node2" #dql_us
)
add_node1 =  make_node(
    "Add", inputs=["out_m1","out_m2"], outputs=["out_add1"],name="add_node1"
)
add_node2 = make_node(
    "Add", inputs=["out_add1","b_s"], outputs=["s_t_ba"],name="add_node2"
)
ql_add = make_node(
    "QuantizeLinear", inputs=["s_t_ba","scale_sigmoid","zero_point_sigmoid"], outputs=["ql_s_t_add"], name="ql_add"
    )
# clp_add = make_node(
#     "Clip", inputs=["ql_s_t_add","min","max"], outputs=["clp_s_t_add"], name="clp_add"
#     )
dql_add = make_node(
    "DequantizeLinear", inputs=["ql_s_t_add","scale_all","zero_point_all"], outputs=["dql_s_t_add"], name="dql_add"
    )
# id_node_1 = make_node(
#     "Identity", inputs=["dql_s_t_add"], outputs=["dql_s_t_add_out"]
# )
sig_s = make_node(
    "Sigmoid", inputs=["dql_s_t_add"], outputs=["s_t"],name="sig_s"
)
ql_act = make_node(
    "QuantizeLinear", inputs=["s_t","scale_sigmoid","zero_point_unsigned"], outputs=["ql_s_t"], name="ql_act"
    )
# clp_act = make_node(
#     "Clip", inputs=["ql_s_t","min","max"], outputs=["clp_s_t"], name="clp_act"
#     )
dql_act = make_node(
    "DequantizeLinear", inputs=["ql_s_t","scale_all","zero_point_unsigned"], outputs=["dql_s_t"], name="dql_act"
    )
id_node_2 = make_node(
    "Identity", inputs=["dql_s_t"], outputs=["scan_out"]
)

bias_val = np.zeros([20,1],dtype=np.float32).reshape([20,1])
Ws_val = np.ones([20,10],dtype=np.float32).reshape([20,10])
Us_val = np.ones([20,20],dtype=np.float32).reshape([20,20])
gen_lstm_eq = onnx.helper.make_graph(
    nodes=[
           ql_w,
           clp_w,
           dql_w,
           ql_u,
           clp_u,
           dql_u, 
           mul_node1, 
           mul_node2, 
           add_node1, 
           add_node2,
           ql_add,
           #clp_add,
           dql_add, 
           #id_node_1,
           sig_s,
           ql_act,
           #clp_act,
           dql_act,
           id_node_2
          ],
    name = "Scan-Body",
    inputs=[inp_h_t_1,inp_X],#The order of the inputs reversed here in order to match the order of inputs of the defined scan node.
    outputs = [scan_out],
    value_info=[
            make_tensor_value_info("out_m1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("s_t_ba",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("ql_ws", onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("dql_ws",onnx.TensorProto.FLOAT, [20,10]),
            make_tensor_value_info("ql_us", onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("dql_us",onnx.TensorProto.FLOAT, [20,20]),
            make_tensor_value_info("ql_s_t_add", onnx.TensorProto.INT8, [20,1]),
            #make_tensor_value_info("clp_s_t_add", onnx.TensorProto.INT8, [20,1]),
            make_tensor_value_info("dql_s_t_add",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("ql_s_t", onnx.TensorProto.UINT8, [20,1]),
            #make_tensor_value_info("clp_s_t", onnx.TensorProto.INT8, [20,1]),
            make_tensor_value_info("dql_s_t",onnx.TensorProto.FLOAT, [20,1])
        ],
    initializer=[make_tensor('W_s',onnx.TensorProto.FLOAT, [20,10], (Ws_val)),
                 make_tensor('U_s',onnx.TensorProto.FLOAT, [20,20], (Us_val)),
                 make_tensor('b_s',onnx.TensorProto.FLOAT, [20,1], (bias_val)),
                 make_tensor('scale_all',onnx.TensorProto.FLOAT,[],[0.00392156862]),
                 make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0]),
                 make_tensor('scale_sigmoid',onnx.TensorProto.FLOAT,[],[0.00392156862]),
                 make_tensor('zero_point_sigmoid',onnx.TensorProto.INT8,[],[0]),
                 make_tensor('zero_point_unsigned',onnx.TensorProto.UINT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 make_tensor('min',onnx.TensorProto.INT8, [],[-128]),
                 make_tensor('max',onnx.TensorProto.INT8, [], [127]),
                 #make_tensor('bitwidth',onnx.TensorProto.INT32, [], [8])
                ]
)

onnx_model = qonnx_make_model(gen_lstm_eq, producer_name="LSTM_eq")
# Have to convert the opset version of the graph here because the clip operator in the previous version did not allow for INT8 inputs.
# It only allowed for FLOAT inputs.
#Converting to opset version '14' to accomodate clip nodes with INT8 and UINT8 input 
onnx_model.opset_import[0].version = 14
#onnx_model_14 = version_converter.convert_version(onnx_model, 14)
onnx.save(onnx_model,'./lstm-gate-sigmoid-unsigned.onnx')

# in_X = np.ones([10,1],dtype=np.float32).reshape([10,1])
in_X = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
in_X = in_X.reshape([10,1])
in_h_t_1 = np.zeros([20,1],dtype=np.float32).reshape([20,1])
input_dict = {}
input_dict["X"] = in_X
input_dict["h_t-1"] = in_h_t_1

sess = rt.InferenceSession(onnx_model.SerializeToString())
output = sess.run(None, input_dict)
print(output[0])
#print(output[1])