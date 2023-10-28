import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from qonnx.util.basic import qonnx_make_model
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper

# Defining the inputs and outputs of the graph we need to create for the graph of the scan body.
# ---------------------------------------------------
# Defining the inputs value info tensors for the compute to be executed for each input.
inp2_m2 = make_tensor_value_info("h_t-1",onnx.TensorProto.FLOAT, [20,1])
inp2_elm1 = make_tensor_value_info("c_t-1", onnx.TensorProto.FLOAT, [20,1])
inp2_m1 = make_tensor_value_info("X",onnx.TensorProto.FLOAT, [10,1])

#Output value info tensor definitions

### Partial output defined for this variable in the graph as we are concatenating the hidden states instead of the values of the output gates.
out_input_forget_matmul = make_tensor_value_info("i_f_matmul", onnx.TensorProto.FLOAT, [20,1])
out_hidden_forget_matmul = make_tensor_value_info("h_f_matmul", onnx.TensorProto.FLOAT, [20,1])
out_hidden_state = make_tensor_value_info("h_t", onnx.TensorProto.FLOAT, [20,1])
out_forget_gate = make_tensor_value_info("f_t_gate", onnx.TensorProto.FLOAT, [20,1])
out_input_gate = make_tensor_value_info("i_t_gate", onnx.TensorProto.FLOAT, [20,1])
out_cell_gate = make_tensor_value_info("c_t_gate", onnx.TensorProto.FLOAT, [20,1])
out_out_gate = make_tensor_value_info("o_t_gate", onnx.TensorProto.FLOAT, [20,1])
out_cell_state = make_tensor_value_info("c_t", onnx.TensorProto.FLOAT, [20,1])
out_hidden_state_concat = make_tensor_value_info("h_t_concat", onnx.TensorProto.FLOAT, [20,1])

#Applying Quantize and Dequantize operation to the input as mentioned in the output onnx graph.
ql_input = make_node("QuantizeLinear", inputs=["X","inp_scale","zero_point_all"], outputs=["ql_input_out"],name="ql_input")
dql_input = make_node("DequantizeLinear", inputs=["ql_input_out", 'inp_scale', "zero_point_all"], outputs=["dql_input_out"],name="dql_input")

#Defining the individual nodes of the graph we want to create.
# So the order in which the inputs are specified matters. Can't describe the inputs in a random order.
# --------------------------------------------
#1st Equation
mul_node1_e1 = make_node("MatMul", inputs=["W_f","dql_input_out"], outputs=["out_m1_e1"], name="mul_node1_e1")
id_node_1_e1 = make_node("Identity", inputs=["out_m1_e1"], outputs=["i_f_matmul"], name="id_node_1_e1")
mul_node2_e1 = make_node("MatMul", inputs=["U_f","h_t-1"], outputs=["out_m2_e1"],name="mul_node2_e1")
id_node_2_e1 = make_node("Identity", inputs=["out_m2_e1"], outputs=["h_f_matmul"], name="id_node_2_e1")
add_node1_e1 = make_node("Add", inputs=["out_m1_e1","out_m2_e1"], outputs=["out_add1_e1"],name="add_node1_e1")
add_node2_e1 = make_node("Add", inputs=["out_add1_e1","b_f"], outputs=["f_t_ba"],name="add_node2_e1")
quant_linear1_e1 = make_node("QuantizeLinear", inputs=["f_t_ba","scale_activations_1","zero_point_all"], outputs=["f_t_ql1"],name="quant_linear1_e1")
dequant_linear1_e1 = make_node("DequantizeLinear", inputs=["f_t_ql1", "scale_activations_1", "zero_point_all"], outputs=["f_t_dql1"], name="dequant_linear1_e1")
sig_f_e1     = make_node("Sigmoid", inputs=["f_t_dql1"], outputs=["f_t"],name="sig_f_e1")
quant_linear2_e1 = make_node("QuantizeLinear", inputs=["f_t","scale_activations_1","zero_point_unsigned"], outputs=["f_t_ql2"],name="quant_linear2_e1")
dequant_linear2_e1 = make_node("DequantizeLinear", inputs=["f_t_ql2", "scale_activations_1", "zero_point_unsigned"], outputs=["f_t_dql2"], name="dequant_linear2_e1")
id_node_3_e1      = make_node("Identity", inputs=["f_t_dql2"], outputs=["f_t_gate"], name="id_node_3_e1")

#2nd Equation
mul_node1_e2 = make_node("MatMul", inputs=["W_i","dql_input_out"], outputs=["out_m1_e2"], name="mul_node1_e2")
mul_node2_e2 = make_node("MatMul", inputs=["U_i","h_t-1"], outputs=["out_m2_e2"],name="mul_node2_e2")
add_node1_e2 = make_node("Add", inputs=["out_m1_e2","out_m2_e2"], outputs=["out_add1_e2"],name="add_node1_e2")
add_node2_e2 = make_node("Add", inputs=["out_add1_e2","b_i"], outputs=["i_t_ba"],name="add_node2_e2")
quant_linear1_e2 = make_node("QuantizeLinear", inputs=["i_t_ba","scale_activations_1","zero_point_all"], outputs=["i_t_ql1"],name="quant_linear1_e2")
dequant_linear1_e2 = make_node("DequantizeLinear", inputs=["i_t_ql1","scale_activations_1", "zero_point_all"], outputs=["i_t_dql1"], name="dequant_linear1_e2")
sig_i_e2     = make_node("Sigmoid", inputs=["i_t_dql1"], outputs=["i_t"],name="sig_i_e2")
quant_linear2_e2 = make_node("QuantizeLinear", inputs=["i_t","scale_activations_1","zero_point_unsigned"], outputs=["i_t_ql2"],name="quant_linear2_e2")
dequant_linear2_e2 = make_node("DequantizeLinear", inputs=["i_t_ql2", "scale_activations_1", "zero_point_unsigned"], outputs=["i_t_dql2"], name="dequant_linear2_e2")
id_node_e2      = make_node("Identity", inputs=["i_t_dql2"], outputs=["i_t_gate"], name="id_node_e2")

#3rd Equation
mul_node1_e3 = make_node("MatMul", inputs=["W_o","dql_input_out"], outputs=["out_m1_e3"], name="mul_node1_e3")
mul_node2_e3 = make_node("MatMul", inputs=["U_o","h_t-1"], outputs=["out_m2_e3"],name="mul_node2_e3")
add_node1_e3 = make_node("Add", inputs=["out_m1_e3","out_m2_e3"], outputs=["out_add1_e3"],name="add_node1_e3")
add_node2_e3 = make_node("Add", inputs=["out_add1_e3","b_o"], outputs=["o_t_ba"],name="add_node2_e3" )
quant_linear1_e3 = make_node("QuantizeLinear", inputs=["o_t_ba","scale_activations_2","zero_point_all"], outputs=["o_t_ql1"],name="quant_linear_e3")
dequant_linear1_e3 = make_node("DequantizeLinear", inputs=["o_t_ql1","scale_activations_2", "zero_point_all"], outputs=["o_t_dql1"], name="dequant_linear_e3")
sig_o_e3     = make_node("Sigmoid", inputs=["o_t_dql1"], outputs=["o_t"],name="sig_o_e3")
quant_linear2_e3 = make_node("QuantizeLinear", inputs=["o_t","scale_activations_2","zero_point_unsigned"], outputs=["o_t_ql2"],name="quant_linear2_e3")
dequant_linear2_e3 = make_node("DequantizeLinear", inputs=["o_t_ql2", "scale_activations_2", "zero_point_unsigned"], outputs=["o_t_dql2"], name="dequant_linear2_e3")
id_node_e3      = make_node("Identity", inputs=["o_t_dql2"], outputs=["o_t_gate"], name="id_node_e3")

#4th Equation
mul_node1_e4 = make_node("MatMul", inputs=["W_c","dql_input_out"], outputs=["out_m1_e4"], name="mul_node1_e4")
mul_node2_e4 = make_node("MatMul", inputs=["U_c","h_t-1"], outputs=["out_m2_e4"],name="mul_node2_e4")
add_node1_e4 = make_node("Add", inputs=["out_m1_e4","out_m2_e4"], outputs=["out_add1_e4"],name="add_node1_e4")
add_node2_e4 = make_node("Add", inputs=["out_add1_e4","b_c"], outputs=["c_t_ba"],name="add_node2_e4")
quant_linear1_e4 = make_node("QuantizeLinear", inputs=["c_t_ba","scale_activations_1","zero_point_all"], outputs=["c_t_ql1"],name="quant_linear1_e4")
dequant_linear1_e4 = make_node("DequantizeLinear", inputs=["c_t_ql1","scale_activations_1", "zero_point_all"], outputs=["c_t_dql1"], name="dequant_linear1_e4")
tanh_c_e4    = make_node("Tanh", inputs=["c_t_dql1"], outputs=["c_t_partial"],name="tanh_c_e4")
quant_linear2_e4 = make_node("QuantizeLinear", inputs=["c_t_partial","scale_activations_1","zero_point_all"], outputs=["c_t_ql2"],name="quant_linear2_e4")
dequant_linear2_e4 = make_node("DequantizeLinear", inputs=["c_t_ql2", "scale_activations_1", "zero_point_all"], outputs=["c_t_dql2"], name="dequant_linear2_e4")
id_node_e4      = make_node("Identity", inputs=["c_t_dql2"], outputs=["c_t_gate"], name="id_node_e4")

#5th Equation
el_mul_node1_e5 = make_node("Mul", inputs=["f_t_dql2","c_t-1"], outputs=["out_el_mul1_e5"],name="el_mul_node1_e5")
quant_linear1_e5 = make_node("QuantizeLinear", inputs=["out_el_mul1_e5","scale_activations_1","zero_point_all"], outputs=["fifth_ql1"],name="quant_linear1_e5")
dequant_linear1_e5 = make_node("DequantizeLinear", inputs=["fifth_ql1","scale_activations_1", "zero_point_all"], outputs=["fifth_dql1"], name="dequant_linear1_e5")
el_mul_node2_e5 = make_node("Mul", inputs=["i_t_dql2","c_t_dql2"], outputs=["out_el_mul2_e5"], name="el_mul_node2_e5") 
quant_linear2_e5 = make_node("QuantizeLinear", inputs=["out_el_mul2_e5","scale_activations_1","zero_point_all"], outputs=["fifth_ql2"],name="quant_linear2_e5")
dequant_linear2_e5 = make_node("DequantizeLinear", inputs=["fifth_ql2","scale_activations_1", "zero_point_all"], outputs=["fifth_dql2"], name="dequant_linear2_e5")
out_add1_e5     = make_node("Add", inputs=["fifth_dql1","fifth_dql2"], outputs=["c_t"], name="out_add1_e5")
quant_linear3_e5 = make_node("QuantizeLinear", inputs=["c_t","scale_activations_1","zero_point_all"], outputs=["h_t_ql"], name="quant_linear3_e5")
dequant_linear3_e5 = make_node("DequantizeLinear", inputs=["h_t_ql","scale_activations_1","zero_point_all"], outputs=["h_t_dql"], name="dequant_linear3_e5")

#6th Equation
tanh_node_e6    = make_node("Tanh", inputs=["h_t_dql"], outputs=["out_tanh_e6"], name="tanh_node_e6") 
quant_linear1_e6 = make_node("QuantizeLinear", inputs=["out_tanh_e6","scale_activations_2","zero_point_all"], outputs=["sixth_ql1"], name="quant_linear1_e6")
dequant_linear1_e6 = make_node("DequantizeLinear", inputs=["sixth_ql1","scale_activations_2","zero_point_all"], outputs=["sixth_dql1"], name="dequant_linear1_e6")
el_mul_node1_e6 = make_node("Mul", inputs=["sixth_dql1","o_t_dql2"], outputs=["h_t_inter"], name="el_mul_node1_e6")#h_t_inter
quant_linear2_e6 = make_node("QuantizeLinear", inputs=["h_t_inter","scale_activations_1","zero_point_all"], outputs=["sixth_ql2"], name="quant_linear2_e6")
dequant_linear2_e6 = make_node("DequantizeLinear", inputs=["sixth_ql2","scale_activations_1","zero_point_all"], outputs=["h_t"], name="dequant_linear2_e6")
id_node_e6      = make_node("Identity", inputs=["h_t"], outputs=["h_t_concat"], name="id_node_e6")
##Adding an Identity node after the hidden state compute to concatenate all the hidden states in the scan node.

qcdq_lstm_no_bias_no_weight_quantization = onnx.load("./quant_lstm_no_bias_no_weight_quantization_qcdq.onnx")
weights = qcdq_lstm_no_bias_no_weight_quantization.graph.initializer
print(len(weights))
for i in range(len(weights)):
    w = numpy_helper.to_array(weights[i])
    print (qcdq_lstm_no_bias_no_weight_quantization.graph.initializer[i].name)
    print(w.shape,',',i)
    print(w)
    print("-------------------------")
    
bi_val = numpy_helper.to_array(weights[0])
Wi_val = numpy_helper.to_array(weights[1])
Ui_val = numpy_helper.to_array(weights[2])

bf_val = numpy_helper.to_array(weights[3])
Wf_val = numpy_helper.to_array(weights[4])
Uf_val = numpy_helper.to_array(weights[5])

bc_val = numpy_helper.to_array(weights[6])
Wc_val = numpy_helper.to_array(weights[7])
Uc_val = numpy_helper.to_array(weights[8])

bo_val = numpy_helper.to_array(weights[9])
Wo_val = numpy_helper.to_array(weights[10])
Uo_val = numpy_helper.to_array(weights[11])

lstm_scan = make_graph(
    nodes=[
           ql_input,
           dql_input, 
           mul_node1_e1, 
           mul_node2_e1, 
           add_node1_e1, 
           add_node2_e1,
           quant_linear1_e1,
           dequant_linear1_e1,
           sig_f_e1,
           quant_linear2_e1,
           dequant_linear2_e1,
           mul_node1_e2, 
           mul_node2_e2, 
           add_node1_e2, 
           add_node2_e2,
           quant_linear1_e2,
           dequant_linear1_e2,
           sig_i_e2,
           quant_linear2_e2,
           dequant_linear2_e2,
           mul_node1_e3, 
           mul_node2_e3, 
           add_node1_e3, 
           add_node2_e3,
           quant_linear1_e3,
           dequant_linear1_e3,
           sig_o_e3,
           quant_linear2_e3,
           dequant_linear2_e3,
           mul_node1_e4, 
           mul_node2_e4, 
           add_node1_e4, 
           add_node2_e4,
           quant_linear1_e4,
           dequant_linear1_e4,
           tanh_c_e4,
           quant_linear2_e4,
           dequant_linear2_e4,
           el_mul_node1_e5,
           quant_linear1_e5,
           dequant_linear1_e5,
           el_mul_node2_e5,
           quant_linear2_e5,
           dequant_linear2_e5,
           out_add1_e5,
           quant_linear3_e5, 
           dequant_linear3_e5,
           tanh_node_e6,
           quant_linear1_e6,
           dequant_linear1_e6,
           el_mul_node1_e6,
           quant_linear2_e6,
           dequant_linear2_e6,   
           id_node_e6
          ],#Can simplify this part by appending the nodes earlier itself.
#     Nodes,
    name = "QCDQ-LSTM-SCAN",
    inputs=[inp2_m2,inp2_elm1,inp2_m1], #The order in which the inputs are defined here should match the input order when the scan node is defined.
    outputs = [out_hidden_state,out_cell_state,out_hidden_state_concat],#out_add2_e3
    value_info=[
            make_tensor_value_info("ql_input_out",onnx.TensorProto.INT8, [10,1]),
            make_tensor_value_info("dql_input_out",onnx.TensorProto.FLOAT, [10,1]),
            make_tensor_value_info("out_m1_e1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("f_t_ba",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("f_t_ql1",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("f_t_dql1", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("f_t_ql2",onnx.TensorProto.UINT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("f_t_dql2", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e2",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("i_t_ba",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("i_t_ql1",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("i_t_dql1", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("i_t_ql2",onnx.TensorProto.UINT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("i_t_dql2", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e3",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e3",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e3",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("o_t_ba",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("o_t_ql1",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("o_t_dql1", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("o_t_ql2",onnx.TensorProto.UINT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("o_t_dql2", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e4",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_m2_e4",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_add1_e4",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("c_t_ba",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("c_t_ql1",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("c_t_dql1", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("c_t_ql2",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("c_t_dql2", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("f_t",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("i_t",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("o_t",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("c_t_partial",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_el_mul1_e5",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("out_el_mul2_e5",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("fifth_ql1",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("fifth_dql1", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("fifth_ql2",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("fifth_dql2", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("h_t_ql",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("h_t_dql", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_tanh_e6",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("sixth_ql1",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("sixth_dql1", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("sixth_ql2",onnx.TensorProto.INT8, [20,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("h_t_inter", onnx.TensorProto.FLOAT, [20,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
        ],
    initializer=[
                 #Scalars 'scale' and 'zero_point' should be defined as below. Converting them into numpy array based single values causes some errors and exceptions saying that these values should be scalar. The definition has to be like this.
                 # Scalars are tensors with undefined shapes.
                 make_tensor('inp_scale',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_activations_1',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_activations_2',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[16]))]),
                 make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 make_tensor('zero_point_unsigned',onnx.TensorProto.UINT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
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
onnx.save(onnx_model, './quantize-lstm-no-bias-no-weight-graph.onnx')

# Defining the values of the varibales to test the execution of the onnx model
in1lstm = np.empty([10,1],dtype=np.float32).reshape([10,1])
in1lstm.fill(10)
in2lstm =  np.zeros((20, 1)).astype(np.float32)
in3lstm =  np.zeros((20, 1)).astype(np.float32)
input_dict = {}
input_dict["X"] = in1lstm
input_dict["h_t-1"] = in2lstm
input_dict["c_t-1"] = in3lstm 

#Executing the onnx model here.
sess = rt.InferenceSession(onnx_model.SerializeToString())
output = sess.run(None, input_dict)
print(output)
print('-------------------------------')

# brevitas_hidden = np.load('./hidden_no_bw.npy')
# print(brevitas_hidden)
# reshaped_output = output[0].reshape([1,20])
# print(reshaped_output)
# print('----------------------------')

# scale_hidden = 0.0078125
# diff_hidden = np.round((reshaped_output - brevitas_hidden)/scale_hidden)
# print('Diff hidden state = ',diff_hidden)

#Defining the input and output value info tensors for the scan_graph creation. These tensors act as the wrapper to the previously defined graph.

#Inputs
scan_input = make_tensor_value_info("scan_input",onnx.TensorProto.FLOAT, [None,10,1])#X ; scan input; Here None defines the varibale number of inputs that can be supplied for input processing.
inp_a      = make_tensor_value_info("inp_a",onnx.TensorProto.FLOAT, [20,1])# h_t-1
inp_b      = make_tensor_value_info("inp_b",onnx.TensorProto.FLOAT, [20,1])# c_t-1

#Outputs
out_a = make_tensor_value_info("out_a", onnx.TensorProto.FLOAT, [20,1])#h_t
out_b = make_tensor_value_info("out_b", onnx.TensorProto.FLOAT, [20,1])#c_t
out_c = make_tensor_value_info("out_c", onnx.TensorProto.FLOAT, [None,20,1])

# Defining the scan node here now
scan_node_lstm = make_node(
    "Scan", 
    inputs=["inp_a","inp_b","scan_input"], 
    outputs=["out_a","out_b","out_c"], 
    num_scan_inputs=1,
    body=lstm_scan, domain=''
)

# Define the graph for the scan node to execute it with onnxruntime.
scan_lstm_node_graph = make_graph(
    nodes = [scan_node_lstm],
    name="lstm-scan-node",
    inputs=[inp_a,inp_b,scan_input],#h_t-1, c_t-1, X
    outputs=[out_a,out_b,out_c]#h_t,c_t,h_t_concat
)

lstm_scan_node_model = qonnx_make_model(scan_lstm_node_graph, producer_name="LSTM-Scan")
onnx.save(lstm_scan_node_model, './lstm_scan_node_model.onnx')

#Checking the model for any errors
onnx.checker.check_model(lstm_scan_node_model)
print(lstm_scan_node_model.graph.value_info)

#Have to convert the opset version of the graph here because the clip operator in the previous version did not allow for INT8 inputs.
# It only allowed for FLOAT inputs.
from onnx import version_converter, helper
lstm_scan_node_model_14 = version_converter.convert_version(lstm_scan_node_model, 14)
# print(lstm_scan_node_model_14)

# Defining the values of the varibales to test the execution of the onnx model
in1_inpa =  np.zeros((20, 1)).astype(np.float32)#'h_t-1'
in2_inpb = np.zeros((20, 1)).astype(np.float32)#'c_t-1'
# in3_scan_input =  np.ones((5, 10, 1)).astype(np.float32)#'X' 10,1 : Because that is the way the shape of the model has been defined.
in3_scan_input = np.empty([25,10,1],dtype=np.float32).reshape([25,10,1])
in3_scan_input.fill(1)
input_dict = {}
input_dict["inp_a"] = in1_inpa
input_dict["inp_b"] = in2_inpb
input_dict["scan_input"] = in3_scan_input

#Executing the onnx model here.
sess = rt.InferenceSession(lstm_scan_node_model_14.SerializeToString())
output = sess.run(None, input_dict)
# print("Final Hidden State = ", output[0].reshape([1,20]))
# print("Final Cell State = ", output[1].reshape([1,20]))
print("All Hidden States = ", output[2].reshape([25,1,20]))

brevitas_out = np.load('./hidden_no_bw.npy')
print("Brevitas hidden states = ",brevitas_out)
# print(brevitas_out[4])
scan_full_out = output[2].reshape([25,1,20])
# print(scan_full_out[4])
comp = scan_full_out - brevitas_out
# print(comp)
scale_hidden = 0.0078125
round_comp = np.round(comp/scale_hidden)
print(round_comp)