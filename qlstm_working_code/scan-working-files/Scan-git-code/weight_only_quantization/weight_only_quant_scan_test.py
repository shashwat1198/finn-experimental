import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from qonnx.util.basic import qonnx_make_model
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper

Inputs = []
Outputs = []
Nodes = []
Initializers = []

# Defining the inputs and outputs of the graph we need to create for the graph of the scan body.
# ---------------------------------------------------
# Defining the inputs value info tensors for the compute to be executed for each input.
inp2_m2 = make_tensor_value_info("h_t-1",onnx.TensorProto.FLOAT, [20,1])
Inputs.append(inp2_m2)
inp2_elm1 = make_tensor_value_info("c_t-1", onnx.TensorProto.FLOAT, [20,1])
Inputs.append(inp2_elm1)
inp2_m1 = make_tensor_value_info("X",onnx.TensorProto.FLOAT, [10,1])
Inputs.append(inp2_m1)

#Output value info tensor definitions

### Partial output defined for this variable in the graph as we are concatenating the hidden states instead of the values of the output gates.
# To concatenate the values of the output date use the below variable.
# out_add2_e3 = make_tensor_value_info("o_t", onnx.TensorProto.FLOAT, [20,1])

out_hidden_state = make_tensor_value_info("h_t", onnx.TensorProto.FLOAT, [20,1])
Outputs.append(out_hidden_state)
out_cell_state = make_tensor_value_info("c_t", onnx.TensorProto.FLOAT, [20,1])
Outputs.append(out_cell_state)
out_hidden_state_concat = make_tensor_value_info("h_t_concat", onnx.TensorProto.FLOAT, [20,1])
Outputs.append(out_hidden_state_concat)

#Pushing the weights quantisation in the scan node.
ql_w1 = make_node("QuantizeLinear", inputs=["W_f","scale_f","zero_point_all"], outputs=["ql_wf_out"], name="ql_w1")
clp_w1 = make_node("Clip", inputs=["ql_wf_out","min","max"], outputs=["clp_wf"], name="clp_w1")
dql_w1 = make_node("DequantizeLinear", inputs=["clp_wf","scale_f","zero_point_all"], outputs=["dql_wf_out"], name="dql_w1")

ql_w2 = make_node("QuantizeLinear", inputs=["W_i","scale_i","zero_point_all"], outputs=["ql_wi_out"], name="ql_w2")
clp_w2 = make_node("Clip", inputs=["ql_wi_out","min","max"], outputs=["clp_wi"], name="clp_w2")
dql_w2 = make_node("DequantizeLinear", inputs=["clp_wi","scale_i","zero_point_all"], outputs=["dql_wi_out"], name="dql_w2")

ql_w3 = make_node("QuantizeLinear", inputs=["W_c","scale_c","zero_point_all"], outputs=["ql_wc_out"], name="ql_w3")
clp_w3 = make_node("Clip", inputs=["ql_wc_out","min","max"], outputs=["clp_wc"], name="clp_w3")
dql_w3 = make_node("DequantizeLinear", inputs=["clp_wc","scale_c","zero_point_all"], outputs=["dql_wc_out"], name="dql_w3")

ql_w4 = make_node("QuantizeLinear", inputs=["W_o","scale_o","zero_point_all"], outputs=["ql_wo_out"], name="ql_w4")
clp_w4 = make_node("Clip", inputs=["ql_wo_out","min","max"], outputs=["clp_wo"], name="clp_w4")
dql_w4 = make_node("DequantizeLinear", inputs=["clp_wo","scale_o","zero_point_all"], outputs=["dql_wo_out"], name="dql_w4")

#These are the quantizations for the recurrence weight matrices.
ql_u1 = make_node("QuantizeLinear", inputs=["U_f","scale_f","zero_point_all"], outputs=["ql_uf_out"], name="ql_u1")
clp_u1 = make_node("Clip", inputs=["ql_uf_out","min","max"], outputs=["clp_uf"], name="clp_u1")
dql_u1 = make_node("DequantizeLinear", inputs=["clp_uf","scale_f","zero_point_all"], outputs=["dql_uf_out"], name="dql_u1")

ql_u2 = make_node("QuantizeLinear", inputs=["U_i","scale_i","zero_point_all"], outputs=["ql_ui_out"], name="ql_u2")
clp_u2 = make_node("Clip", inputs=["ql_ui_out","min","max"], outputs=["clp_ui"], name="clp_u2")
dql_u2 = make_node("DequantizeLinear", inputs=["clp_ui","scale_i","zero_point_all"], outputs=["dql_ui_out"], name="dql_u2")

ql_u3 = make_node("QuantizeLinear", inputs=["U_c","scale_c","zero_point_all"], outputs=["ql_uc_out"], name="ql_u3")
clp_u3 = make_node("Clip", inputs=["ql_uc_out","min","max"], outputs=["clp_uc"], name="clp_u3")
dql_u3 = make_node("DequantizeLinear", inputs=["clp_uc","scale_c","zero_point_all"], outputs=["dql_uc_out"], name="dql_u3")

ql_u4 = make_node("QuantizeLinear", inputs=["U_o","scale_o","zero_point_all"], outputs=["ql_uo_out"], name="ql_u4")
clp_u4 = make_node("Clip", inputs=["ql_uo_out","min","max"], outputs=["clp_uo"], name="clp_u4")
dql_u4 = make_node("DequantizeLinear", inputs=["clp_uo","scale_o","zero_point_all"], outputs=["dql_uo_out"], name="dql_u4")

#Defining the individual nodes of the graph we want to create.
# So the order in which the inputs are specified matters. Can't describe the inputs in a random order.
# --------------------------------------------
#1st Equation
mul_node1_e1 = make_node("MatMul", inputs=["dql_wf_out","X"], outputs=["out_m1_e1"], name="mul_node1_e1")
Nodes.append(mul_node1_e1)
mul_node2_e1 = make_node("MatMul", inputs=["dql_uf_out","h_t-1"], outputs=["out_m2_e1"],name="mul_node2_e1")
Nodes.append(mul_node2_e1)
add_node1_e1 = make_node("Add", inputs=["out_m1_e1","out_m2_e1"], outputs=["out_add1_e1"],name="add_node1_e1")
Nodes.append(add_node1_e1)
add_node2_e1 = make_node("Add", inputs=["out_add1_e1","b_f"], outputs=["f_t_ba"],name="add_node2_e1")
Nodes.append(add_node2_e1)
sig_f_e1     = make_node("Sigmoid", inputs=["f_t_ba"], outputs=["f_t"],name="sig_f_e1")
Nodes.append(sig_f_e1)

#2nd Equation
mul_node1_e2 = make_node("MatMul", inputs=["dql_wi_out","X"], outputs=["out_m1_e2"], name="mul_node1_e2")
Nodes.append(mul_node1_e2)
mul_node2_e2 = make_node("MatMul", inputs=["dql_ui_out","h_t-1"], outputs=["out_m2_e2"],name="mul_node2_e2")
Nodes.append(mul_node2_e2)
add_node1_e2 = make_node("Add", inputs=["out_m1_e2","out_m2_e2"], outputs=["out_add1_e2"],name="add_node1_e2")
Nodes.append(add_node1_e2)
add_node2_e2 = make_node("Add", inputs=["out_add1_e2","b_i"], outputs=["i_t_ba"],name="add_node2_e2")
Nodes.append(add_node2_e2)
sig_i_e2     = make_node("Sigmoid", inputs=["i_t_ba"], outputs=["i_t"],name="sig_i_e2")
Nodes.append(sig_i_e2)

#3rd Equation
mul_node1_e3 = make_node("MatMul", inputs=["dql_wo_out","X"], outputs=["out_m1_e3"], name="mul_node1_e3")
Nodes.append(mul_node1_e3)
mul_node2_e3 = make_node("MatMul", inputs=["dql_uo_out","h_t-1"], outputs=["out_m2_e3"],name="mul_node2_e3")
Nodes.append(mul_node2_e3)
add_node1_e3 = make_node("Add", inputs=["out_m1_e3","out_m2_e3"], outputs=["out_add1_e3"],name="add_node1_e3")
Nodes.append(add_node1_e3)
add_node2_e3 = make_node("Add", inputs=["out_add1_e3","b_o"], outputs=["o_t_ba"],name="add_node2_e3" )
Nodes.append(add_node2_e3)
sig_o_e3     = make_node("Sigmoid", inputs=["o_t_ba"], outputs=["o_t"],name="sig_o_e3")
Nodes.append(sig_o_e3)

#4th Equation
mul_node1_e4 = make_node("MatMul", inputs=["dql_wc_out","X"], outputs=["out_m1_e4"], name="mul_node1_e4")
Nodes.append(mul_node1_e4)
mul_node2_e4 = make_node("MatMul", inputs=["dql_uc_out","h_t-1"], outputs=["out_m2_e4"],name="mul_node2_e4")
Nodes.append(mul_node2_e4)
add_node1_e4 = make_node("Add", inputs=["out_m1_e4","out_m2_e4"], outputs=["out_add1_e4"],name="add_node1_e4")
Nodes.append(add_node1_e4)
add_node2_e4 = make_node("Add", inputs=["out_add1_e4","b_c"], outputs=["c_t_ba"],name="add_node2_e4")
Nodes.append(add_node2_e4)
tanh_c_e4    = make_node("Tanh", inputs=["c_t_ba"], outputs=["c_t_partial"],name="tanh_c_e4")
Nodes.append(tanh_c_e4)

#5th Equation
el_mul_node1_e5 = make_node("Mul", inputs=["f_t","c_t-1"], outputs=["out_el_mul1_e5"],name="el_mul_node1_e5")
Nodes.append(el_mul_node1_e5)
el_mul_node2_e5 = make_node("Mul", inputs=["i_t","c_t_partial"], outputs=["out_el_mul2_e5"], name="el_mul_node2_e5") 
Nodes.append(el_mul_node2_e5)
out_add1_e5     = make_node("Add", inputs=["out_el_mul1_e5","out_el_mul2_e5"], outputs=["c_t"], name="out_add1_e5")
Nodes.append(out_add1_e5)

#6th Equation
tanh_node_e6    = make_node("Tanh", inputs=["c_t"], outputs=["out_tanh_e6"], name="tanh_node_e6") 
Nodes.append(tanh_node_e6)
el_mul_node1_e6 = make_node("Mul", inputs=["out_tanh_e6","o_t"], outputs=["h_t"], name="el_mul_node1_e6")
Nodes.append(el_mul_node1_e6)
id_node_e6      = make_node("Identity", inputs=["h_t"], outputs=["h_t_concat"], name="id_node_e6")
Nodes.append(id_node_e6)
##Adding an Identity node after the hidden state compute, to concatenate all the hidden states in the scan node.

qcdq_lstm_weight_only = onnx.load("./quant_lstm_weight_only_4b_25.onnx")
weights = qcdq_lstm_weight_only.graph.initializer
# print(weights[0].shape)
print(len(weights))
for i in range(len(weights)):
    w = numpy_helper.to_array(weights[i])
    print (qcdq_lstm_weight_only.graph.initializer[i].name)
    print(w.shape,',',i)
    print(w)
    print("-------------------------")

# print (qcdq_lstm_weight_only.graph.node[0].input[1]) # dense_input         = 1. layer
# print (qcdq_lstm_weight_only.graph.initializer[0].name) # dense_1/kernel:0 = last layer

#Order in which to read the weights is = Input, forget, cell and output. Got this from the initializer names.
Wi_val = numpy_helper.to_array(weights[0])
Ui_val = numpy_helper.to_array(weights[1])
Wf_val = numpy_helper.to_array(weights[2])
Uf_val = numpy_helper.to_array(weights[3])
Wc_val = numpy_helper.to_array(weights[4])
Uc_val = numpy_helper.to_array(weights[5])
Wo_val = numpy_helper.to_array(weights[6])
Uo_val = numpy_helper.to_array(weights[7])

all_bias = numpy_helper.to_array(weights[8])
all_bias = all_bias.reshape([160,1])

#Order in which to read the biases = Input, forget, output and cell. Gives the best results. But could not get this information from the intitilaizer names.
#Tried this random order out of the available 24 and the above ones worked properly.
#This order of bias values is very important. If we don't follow this order and change it to i,f,c,o as in the case of weights, the outputs don't match and start giving substantial difference between values! 

bi_val = all_bias[0:20,:]
bf_val = all_bias[20:40,:]
bo_val = all_bias[40:60,:] # Changed this from o to c to test
bc_val = all_bias[60:80,:] # So the biases maybe in this order. When I read the bias values in this order are same upto the second decimal place.
#I have been able to test the above bias order with np.random.uniform([5,20,1]) inputs.
#And the results are very close. There is some differnce that is coming up! It maybe because of the way the weights are unsqueezd and concatenated maybe some values change there.

# print(bi_val.shape)
# print(onnx_model)

# So all the weights and biases are stored as initializers in the onnx graph. We access them and then print the shapes of the each of them.
# So the first 8 entries in the weights stored as initialzed correspond to the 4 paris of weight and recurrence matrix(W_s, U_s) as evident from their shapes.
# Weight Matrix : [Output_Dimension X Input_Dimension], Recurrence Matrix : [Output_Dimension x Output_Dimension]
# The last entry corresponds to the concatenated weights and recurrence biases (Total 8 each with shape : [Output_Dimension x 1]).
# Hence, in our case the final shape is [1,160]. From the values we also see that the values of the recurrent biases currently are set to '0'.

#Questions : 
#1. I can access the weights values in the onnx graph. But they don't have scale_factor and zero_point in their values. So these are not quantized values I am assuming. Which will bring out different outputs when executed. 
#2. What are the initial values of the hidden and cell state used? 'initial_h' and 'initial_c' are also variables and if they are not initialized then they are considered '0' in the original LSTM onnx cell. So I am assuming the same here.
#3. How are the quantized inputs fed into the model? But for this case with only weight quantization; inputs are not quantized and are fed directly to the model.

#The outputs are in the order of all concatenated hidden states, final hidden state, final cell state.
#Will have to edit the scan-body of the scan node as it looks that the all the weight and recurrence matrices are concatenated before they are fed into the custom_op.

print('----------- Serializing the onnxruntime session ---------')
sess = rt.InferenceSession(qcdq_lstm_weight_only.SerializeToString())
input_name = sess.get_inputs()[0].name
# print(input_name)
in1_qcdq = np.empty([25,1,10],dtype=np.float32)
in1_qcdq.fill(0.25) #Input to the brevitas exported model
# print(in1_qcdq.shape)
pred_onnx = sess.run(None, {input_name: in1_qcdq})
print('Brevitas All Hidden States : ', pred_onnx[0])

print('-------------- Starting the construction of the body of SCAN graph -----------')
lstm_scan = make_graph(
    nodes=[
           ql_w1,
           clp_w1, 
           dql_w1,
           ql_w2,
           clp_w2, 
           dql_w2,
           ql_w3,
           clp_w3, 
           dql_w3,
           ql_w4,
           clp_w4, 
           dql_w4,
           ql_u1,
           clp_u1, 
           dql_u1,
           ql_u2,
           clp_u2,
           dql_u2,    
           ql_u3,
           clp_u3,
           dql_u3,    
           ql_u4,
           clp_u4,
           dql_u4, 
           mul_node1_e1, 
           mul_node2_e1, 
           add_node1_e1, 
           add_node2_e1,
           sig_f_e1,
           mul_node1_e2, 
           mul_node2_e2, 
           add_node1_e2, 
           add_node2_e2,
           sig_i_e2,
           mul_node1_e3, 
           mul_node2_e3, 
           add_node1_e3, 
           add_node2_e3,
           sig_o_e3,
           mul_node1_e4, 
           mul_node2_e4, 
           add_node1_e4, 
           add_node2_e4,
           tanh_c_e4,
           el_mul_node1_e5,
           el_mul_node2_e5,
           out_add1_e5,
           tanh_node_e6,
           el_mul_node1_e6,   
           id_node_e6
          ],#Can simplify this part by appending the nodes earlier itself.
#     Nodes,
    name = "QCDQ-LSTM-SCAN",
    inputs=[inp2_m2,inp2_elm1,inp2_m1], #The order in which the inputs are defined here should match the input order when the scan node is defined.
    outputs = [out_hidden_state,out_cell_state,out_hidden_state_concat],#out_add2_e3
    value_info=[
            make_tensor_value_info("out_m1_e1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("f_t_ba",onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("i_t_ba",onnx.TensorProto.FLOAT, [20,1]),#output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e3",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e3",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e3",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("o_t_ba",onnx.TensorProto.FLOAT, [20,1]),#output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e4",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e4",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e4",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("c_t_ba",onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("f_t",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("i_t",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("o_t",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("c_t_partial",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_el_mul1_e5",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_el_mul2_e5",onnx.TensorProto.FLOAT, [20,1]),# output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_tanh_e6",onnx.TensorProto.FLOAT, [20,1]),#output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("ql_wf_out", onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("dql_wf_out",onnx.TensorProto.FLOAT, [20,10]),
            make_tensor_value_info("ql_wi_out", onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("dql_wi_out",onnx.TensorProto.FLOAT, [20,10]),
            make_tensor_value_info("ql_wc_out", onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("dql_wc_out",onnx.TensorProto.FLOAT, [20,10]),
            make_tensor_value_info("ql_wo_out", onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("dql_wo_out",onnx.TensorProto.FLOAT, [20,10]),
            make_tensor_value_info("ql_uf_out",onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("dql_uf_out",onnx.TensorProto.FLOAT, [20,20]),
            make_tensor_value_info("ql_ui_out",onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("dql_ui_out",onnx.TensorProto.FLOAT, [20,20]),
            make_tensor_value_info("ql_uc_out",onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("dql_uc_out",onnx.TensorProto.FLOAT, [20,20]),
            make_tensor_value_info("ql_uo_out",onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("dql_uo_out",onnx.TensorProto.FLOAT, [20,20]),
            make_tensor_value_info("clp_wf",onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("clp_wi",onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("clp_wc",onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("clp_wo",onnx.TensorProto.INT8, [20,10]),
            make_tensor_value_info("clp_uf",onnx.TensorProto.INT8, [20,20]), 
            make_tensor_value_info("clp_ui",onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("clp_uc",onnx.TensorProto.INT8, [20,20]),
            make_tensor_value_info("clp_uo",onnx.TensorProto.INT8, [20,20])
        ],
    initializer=[#Scalars 'scale' and 'zero_point' should be defined as below. Converting them into numpy array based single values causes some errors and exceptions saying that these values should be scalar. The definition has to be like this.
                 # Scalars are tensors with undefined shapes.
                 make_tensor('scale_i',onnx.TensorProto.FLOAT,[],[float(numpy_helper.to_array(weights[9]))]),
                 make_tensor('scale_c',onnx.TensorProto.FLOAT,[],[float(numpy_helper.to_array(weights[13]))]),
                 make_tensor('scale_o',onnx.TensorProto.FLOAT,[],[float(numpy_helper.to_array(weights[14]))]),
                 make_tensor('scale_f',onnx.TensorProto.FLOAT,[],[float(numpy_helper.to_array(weights[15]))]),
                 make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 #Introducing scalars for the clip operators.
                 make_tensor('min', onnx.TensorProto.INT8, [], [-7]), #Refers to 8-bit quantization
                 make_tensor('max', onnx.TensorProto.INT8, [], [7]),
                 make_tensor('W_f',onnx.TensorProto.FLOAT, [20,10], (Wf_val)),
                 make_tensor('U_f',onnx.TensorProto.FLOAT, [20,20], (Uf_val)),
                 make_tensor('b_f',onnx.TensorProto.FLOAT, [20,1], (bf_val)),
                 make_tensor('W_i',onnx.TensorProto.FLOAT, [20,10], (Wi_val)),
                 make_tensor('U_i',onnx.TensorProto.FLOAT, [20,20], (Ui_val)),
                 make_tensor('b_i',onnx.TensorProto.FLOAT, [20,1], (bi_val)),
                 make_tensor('W_o',onnx.TensorProto.FLOAT, [20,10], (Wo_val)),
                 make_tensor('U_o',onnx.TensorProto.FLOAT, [20,20], (Uo_val)),
                 make_tensor('b_o',onnx.TensorProto.FLOAT, [20,1], (bo_val)),
                 make_tensor('W_c',onnx.TensorProto.FLOAT, [20,10], (Wc_val)),
                 make_tensor('U_c',onnx.TensorProto.FLOAT, [20,20], (Uc_val)),
                 make_tensor('b_c',onnx.TensorProto.FLOAT, [20,1], (bc_val))
                ]
)

onnx_model = qonnx_make_model(lstm_scan, producer_name="QuantizeLSTM_scan")
print('-------------- Construction of the body of SCAN graph complete -----------')

print('--------------- Converting the onnx model to version 14 to accomodate the clip operation ------------')
from onnx import version_converter, helper
onnx_model_14 = version_converter.convert_version(onnx_model, 14)
print('--------------- ONNX model converted to version 14 ----------------------')
# Defining the values of the varibales to test the execution of the onnx model
# in1lstm =  np.ones((10, 1)).astype(np.float32)
in1lstm = np.empty([10,1],dtype=np.float32)
in1lstm.fill(0.25)
# print(in1_qcdq[0].shape)
# in1lstm = in1_qcdq[0].reshape([10,1])
in2lstm =  np.zeros((20, 1)).astype(np.float32)
in3lstm =  np.zeros((20, 1)).astype(np.float32)
input_dict = {}
input_dict["X"] = in1lstm            # Input
input_dict["h_t-1"] = in2lstm        # Initial hidden state
input_dict["c_t-1"] = in3lstm        # Initial cell state

sess = rt.InferenceSession(onnx_model_14.SerializeToString())
output = sess.run(None, input_dict)
# print(output)
# print('---------------------------')

#Defining the input and output value info tensors for the scan_graph creation. These tensors act as the wrapper to the previously defined graph.
print('----------Started construction of SCAN node with the above defined body-----------------')
#Inputs
scan_input = make_tensor_value_info("scan_input",onnx.TensorProto.FLOAT, [None,10,1])#X ; scan input; Here None defines the varibale number of inputs that can be supplied for input processing.
inp_a      = make_tensor_value_info("inp_a",onnx.TensorProto.FLOAT, [20,1])# h_t-1
inp_b      = make_tensor_value_info("inp_b",onnx.TensorProto.FLOAT, [20,1])# c_t-1

#Outputs
out_a = make_tensor_value_info("out_a", onnx.TensorProto.FLOAT, [20,1])#h_t
out_b = make_tensor_value_info("out_b", onnx.TensorProto.FLOAT, [20,1])#c_t
out_c = make_tensor_value_info("out_c", onnx.TensorProto.FLOAT, [None,20,1])
#This can be 'o_t' and it can also concatenate the outputs of the intermediate hidden states. 
#In the onnx LSTM cell all the hidden states are concatenated and given as outputs. So to match that I am doing the same.
#For constants can define a constant tensor here. Maybe that will help solve this issue.
#Both the scan input and the scan output have the None shape as the first dimension of the tensor. This allows the execution of unknown outputs and removes all the warnings when the graph is serialized.

# Defining the scan node here now
scan_node_lstm = make_node(
    "Scan", 
    inputs=["inp_a","inp_b","scan_input"], 
    outputs=["out_a","out_b","out_c"], 
    num_scan_inputs=1,
    body=lstm_scan, domain=''
)
# The order in which the nodes are defined in the inputs and outputs also matter here and should match the order defined in the body graph.

# Define the graph for the scan node to execute it with onnxruntime.
scan_lstm_node_graph = make_graph(
    nodes = [scan_node_lstm],
    name="lstm-scan-node",
    inputs=[inp_a,inp_b,scan_input],#h_t-1, c_t-1, X
    outputs=[out_a,out_b,out_c]#h_t,c_t,h_t_concat
)
#Here, the scan input is 'scan_input' connected -> X in the compute graph. This will contain the input data that needs to be processed.
#The scan output is 'out_c' connected -> h_t_concat in the compute graph. out_a and out_b are connected to h_t-1 and c_t-1 in the compute graph and keep getting updated after each input is processed.

lstm_scan_node_model = qonnx_make_model(scan_lstm_node_graph, producer_name="LSTM-Scan")
onnx.save(lstm_scan_node_model, './lstm_scan_node_model.onnx')

#Checking the model for any errors
onnx.checker.check_model(lstm_scan_node_model)
print(lstm_scan_node_model.graph.value_info)

print('----------Construction of SCAN node completed-----------------')

#Have to convert the opset version of the graph here because the clip operator in the previous version did not allow for INT8 inputs.
# It only allowed for FLOAT inputs.
from onnx import version_converter, helper
lstm_scan_node_model_14 = version_converter.convert_version(lstm_scan_node_model, 14)
# print(lstm_scan_node_model_14)

# Defining the values of the varibales to test the execution of the onnx model
in1_inpa =  np.zeros((20, 1)).astype(np.float32)#'h_t-1'
in2_inpb = np.zeros((20, 1)).astype(np.float32)#'c_t-1'
# in3_scan_input =  np.ones((5, 10, 1)).astype(np.float32)#'X' 10,1 : Because that is the way the shape of the model has been defined.
in3_scan_input = in1_qcdq.reshape([25,10,1])
input_dict = {}
input_dict["inp_a"] = in1_inpa
input_dict["inp_b"] = in2_inpb
input_dict["scan_input"] = in3_scan_input

#Executing the onnx model here.
sess = rt.InferenceSession(lstm_scan_node_model_14.SerializeToString())
output = sess.run(None, input_dict)
# print("Final Hidden State = ", output[0])
# print("Final Cell State = ", output[1])
print("All Hidden States = ", output[2].reshape([25,1,20]))
reshaped_output = output[2].reshape([25,1,20])

##Covert to int comparison
#scale_hidden = 0.03192721
scale_hidden = numpy_helper.to_array(weights[9]) # A higher value of this scale reduces the error very much
diff_hidden = np.round((reshaped_output - pred_onnx[0])/scale_hidden)
print('Diff hidden state = ',diff_hidden)