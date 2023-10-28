import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from qonnx.util.basic import qonnx_make_model
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_tensor

## Quantizing weights with the 'Quant' node in QONNX

quant_w = onnx.helper.make_node(
    'Quant',
    domain='finn.custom_op.general',
    inputs=['W_s', 'scale_all', 'zero_point_all', 'bitwidth'],
    outputs=['quant_ws'],
    narrow=0,
    signed=1,
    rounding_mode="ROUND",
)

quant_u = onnx.helper.make_node(
    'Quant',
    domain='finn.custom_op.general',
    inputs=['U_s', 'scale_all', 'zero_point_all', 'bitwidth'],
    outputs=['quant_us'],
    narrow=0,
    signed=1,
    rounding_mode="ROUND",
)

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

#Defining the individual nodes of the graph we want to create.
# --------------------------------------------
mul_node1 = make_node(
"MatMul", inputs=["quant_ws","X"], outputs=["out_m1"], name="mul_node1"
)

mul_node2 = make_node(
"MatMul", inputs=["quant_us","h_t-1"], outputs=["out_m2"],name="mul_node2"
)

add_node1 =  make_node(
"Add", inputs=["out_m1","out_m2"], outputs=["out_add1"],name="add_node1"
)

add_node2 = make_node(
"Add", inputs=["out_add1","b_s"], outputs=["s_t_ba"],name="add_node2"
)

sig_s = make_node(
"Sigmoid", inputs=["s_t_ba"], outputs=["s_t"],name="sig_s"
)

id_node = make_node(
"Identity", inputs=["s_t"], outputs=["scan_out"]
)

bias_val = np.ones([20,1],dtype=np.float32).reshape([20,1])
Ws_val = np.ones([20,10],dtype=np.float32).reshape([20,10])
Us_val = np.ones([20,20],dtype=np.float32).reshape([20,20])
quant_lstm_eq = onnx.helper.make_graph(
    nodes=[
           quant_w,
           quant_u, 
           mul_node1, 
           mul_node2, 
           add_node1, 
           add_node2,
           sig_s,
           id_node
          ],
    name = "Scan-Body",
    inputs=[inp_h_t_1,inp_X],#The order of the inputs reversed here in order to match the order of inputs of the defined scan node.
    outputs = [out_state, scan_out],
    value_info=[
            make_tensor_value_info("out_m1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("s_t_ba",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("quant_ws", onnx.TensorProto.FLOAT, [20,10]),
            make_tensor_value_info("quant_us",onnx.TensorProto.FLOAT, [20,20])
        ],
    initializer=[make_tensor('W_s',onnx.TensorProto.FLOAT, [20,10], (Ws_val)),
                 make_tensor('U_s',onnx.TensorProto.FLOAT, [20,20], (Us_val)),
                 make_tensor('b_s',onnx.TensorProto.FLOAT, [20,1], (bias_val)),
                 make_tensor('scale_all',onnx.TensorProto.FLOAT,[],[1]),
                 make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0]),
                 make_tensor('bitwidth',onnx.TensorProto.INT32, [], [4])
                ]
)
#So some points to note here:
#1. Initializers ('W_s','U_s' and 'b_s') are a part of the model and they will not be defined in the list of the inputs.
#2. Because they are a part of the model, these initializers will not be defined again when we define the scan node later, which we will see.
#3. Scan node only cares about the inputs and outputs of the body_graph and does not care what happens inside it.

onnx_model = qonnx_make_model(quant_lstm_eq, producer_name="LSTM_eq")
onnx.save(onnx_model, './quant_lstm_eq.onnx')

print('---Onnx graph saved with the name : quant_lstm_eq.onnx---')
print('Starting the execution of the graph with the qonnx runtime.......')

## Using 'ModelWrapper' from qonnx to execute this graph with qonnx.

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

model = ModelWrapper("./quant_lstm_eq.onnx")
# print(model)
in_X = np.asarray(np.random.randint(low=0, high=1, size=(10,1)), dtype=np.float32)
in_h_t_1 = np.asarray(np.random.randint(low=0, high=1, size=(20,1)), dtype=np.float32)
np.save('./in0.npy', in_X)
np.save('./in1.npy', in_h_t_1)
idict = {"X" : np.load("./in0.npy"), "h_t-1" : np.load("./in1.npy")}
output = execute_onnx(input_dict=idict,model=model)
print(output)

print('---Execution with the onnx runtime completed---')

## Part 2 : Now checking if a graph with a 'Quant' node {non-standard onnx node} can be imported into the 'body' attribute of the scan node.

#Defining the input and output value info tensors for the scan_graph creation. These tensors act as the wrapper to the previously defined graph.
print('---Importing the above graph in the scan node---')
#Inputs
scan_input = make_tensor_value_info(
"scan_input",onnx.TensorProto.FLOAT, [None,10,1]
)#X ; scan input; Here 'None' defines the varibale number of inputs that can be supplied for input processing.

inp_a = make_tensor_value_info(
"inp_a",onnx.TensorProto.FLOAT, [20,1]
)# h_t-1

#Outputs
out_a = make_tensor_value_info(
"out_a", onnx.TensorProto.FLOAT, [20,1]
)#s_t

out_b = make_tensor_value_info(
"out_b", onnx.TensorProto.FLOAT, [None,20,1]
)#scan_out

# Defining the scan node here now
scan_node_gen_lstm_eq = make_node(
    "Scan", inputs=["inp_a","scan_input"], 
    outputs=["out_a","out_b"], 
    num_scan_inputs=1,
    body=quant_lstm_eq, domain=''
)# The order in which the nodes are defined in the inputs and outputs also matter here.

gen_lstm_scan_graph = make_graph(
    nodes = [scan_node_gen_lstm_eq],
    name="gen_eq_graph",
    inputs=[inp_a,scan_input],
    outputs=[out_a,out_b]
)

scan_quant_model = qonnx_make_model(gen_lstm_scan_graph, producer_name="eq-model")
onnx.save(scan_quant_model, './scan_quant_lstm_eq.onnx')
print('---Onnx graph saved with the name : scan_quant_lstm_eq.onnx---')

scan_model = ModelWrapper("./scan_quant_lstm_eq.onnx")
print('---Printing the scan node---')
print(scan_model)
print('---Starting scan node execution with qonnx runtime--- : Will not execute as  qonnx does not know about scan node, It passes the execution to general onnxruntime and the onnxruntime does not know about the quant node and throws the below error saying bad node spec.')
n = 0
scan_inp_X = np.asarray(np.random.randint(low=0, high=1, size=(n,10,1)), dtype=np.float32)
scan_inp_h_t_1 = np.asarray(np.random.randint(low=0, high=1, size=(20,1)), dtype=np.float32)
np.save('./scan_in0.npy', scan_inp_X)
np.save('./scan_in1.npy', scan_inp_h_t_1)
scan_idict = {"scan_input" : np.load("./scan_in0.npy"), "inp_a" : np.load("./scan_in1.npy")}
scan_output = execute_onnx(input_dict=scan_idict,model=scan_model)
print(scan_output)