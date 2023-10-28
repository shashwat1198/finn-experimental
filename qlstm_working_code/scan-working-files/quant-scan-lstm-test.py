import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
# from finn.util.visualization import showInNetron
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
# out_first_input_out = make_tensor_value_info("first_input_out", onnx.TensorProto.INT8, [10,1])
# out_input_out = make_tensor_value_info("input_out", onnx.TensorProto.FLOAT, [10,1])

# out_ii = make_tensor_value_info("ii", onnx.TensorProto.FLOAT, [20,1])
# out_hi = make_tensor_value_info("hi", onnx.TensorProto.FLOAT, [20,1])
# out_input_acc = make_tensor_value_info("input_acc", onnx.TensorProto.FLOAT, [20,1])
# out_input_gate = make_tensor_value_info("i_t_gate", onnx.TensorProto.FLOAT, [20,1])

# out_if = make_tensor_value_info("if", onnx.TensorProto.FLOAT, [20,1])
# out_hf = make_tensor_value_info("hf", onnx.TensorProto.FLOAT, [20,1])
# out_forget_acc = make_tensor_value_info("forget_acc", onnx.TensorProto.FLOAT, [20,1])
# out_forget_gate = make_tensor_value_info("f_t_gate", onnx.TensorProto.FLOAT, [20,1])

# out_ic = make_tensor_value_info("ic", onnx.TensorProto.FLOAT, [20,1])
# out_hc = make_tensor_value_info("hc", onnx.TensorProto.FLOAT, [20,1])
# out_cell_acc = make_tensor_value_info("cell_acc", onnx.TensorProto.FLOAT, [20,1])
# out_cell_gate = make_tensor_value_info("c_t_gate", onnx.TensorProto.FLOAT, [20,1])

# out_io = make_tensor_value_info("io", onnx.TensorProto.FLOAT, [20,1])
# out_ho = make_tensor_value_info("ho", onnx.TensorProto.FLOAT, [20,1])
# out_out_acc = make_tensor_value_info("out_acc", onnx.TensorProto.FLOAT, [20,1])
# out_out_gate = make_tensor_value_info("o_t_gate", onnx.TensorProto.FLOAT, [20,1])

out_hidden_state = make_tensor_value_info("h_t", onnx.TensorProto.FLOAT, [20,1])
out_cell_state = make_tensor_value_info("c_t", onnx.TensorProto.FLOAT, [20,1])
out_hidden_state_concat = make_tensor_value_info("h_t_concat", onnx.TensorProto.FLOAT, [20,1])

#Applying Quantize and Dequantize operation to the input as mentioned in the output onnx graph.
ql_input = make_node("QuantizeLinear", inputs=["X","inp_scale","zero_point_all"], outputs=["ql_input_out"],name="ql_input")
id_node_0_e0 = make_node("Identity", inputs=["ql_input_out"], outputs=["first_input_out"], name="id_node_0_e0")
dql_input = make_node("DequantizeLinear", inputs=["ql_input_out", 'inp_scale', "zero_point_all"], outputs=["dql_input_out"],name="dql_input")
id_node_1_e0 = make_node("Identity", inputs=["dql_input_out"], outputs=["input_out"], name="id_node_1_e0")

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

#1st Equation : Forget gate
mul_node1_e1 = make_node("MatMul", inputs=["dql_wf_out","dql_input_out"], outputs=["out_m1_e1"], name="mul_node1_e1")
# id_node_1_e1 = make_node("Identity", inputs=["out_m1_e1"], outputs=["if"], name="id_node_1_e1")
mul_node2_e1 = make_node("MatMul", inputs=["dql_uf_out","h_t-1"], outputs=["out_m2_e1"],name="mul_node2_e1")
# id_node_2_e1 = make_node("Identity", inputs=["out_m2_e1"], outputs=["hf"], name="id_node_2_e1")
add_node1_e1 = make_node("Add", inputs=["out_m1_e1","out_m2_e1"], outputs=["out_add1_e1"],name="add_node1_e1")
add_node2_e1 = make_node("Add", inputs=["out_add1_e1","b_f"], outputs=["f_t_ba"],name="add_node2_e1")
quant_linear1_e1 = make_node("QuantizeLinear", inputs=["f_t_ba","scale_3","zero_point_all"], outputs=["f_t_ql1"],name="quant_linear1_e1")
dequant_linear1_e1 = make_node("DequantizeLinear", inputs=["f_t_ql1", "scale_4", "zero_point_all"], outputs=["f_t_dql1"], name="dequant_linear1_e1")
# id_node_3_e1      = make_node("Identity", inputs=["f_t_dql1"], outputs=["forget_acc"], name="id_node_3_e1")
sig_f_e1     = make_node("Sigmoid", inputs=["f_t_dql1"], outputs=["f_t"],name="sig_f_e1")
quant_linear2_e1 = make_node("QuantizeLinear", inputs=["f_t","scale_4","zero_point_unsigned"], outputs=["f_t_ql2"],name="quant_linear2_e1")
dequant_linear2_e1 = make_node("DequantizeLinear", inputs=["f_t_ql2", "scale_4", "zero_point_unsigned"], outputs=["f_t_dql2"], name="dequant_linear2_e1")
# id_node_4_e1      = make_node("Identity", inputs=["f_t_dql2"], outputs=["f_t_gate"], name="id_node_4_e1")

#2nd Equation : Input gate
mul_node1_e2 = make_node("MatMul", inputs=["dql_wi_out","dql_input_out"], outputs=["out_m1_e2"], name="mul_node1_e2")
# id_node_1_e2 = make_node("Identity", inputs=["out_m1_e2"], outputs=["ii"], name="id_node_1_e2")
mul_node2_e2 = make_node("MatMul", inputs=["dql_ui_out","h_t-1"], outputs=["out_m2_e2"],name="mul_node2_e2")
# id_node_2_e2 = make_node("Identity", inputs=["out_m2_e2"], outputs=["hi"], name="id_node_2_e2")
add_node1_e2 = make_node("Add", inputs=["out_m1_e2","out_m2_e2"], outputs=["out_add1_e2"],name="add_node1_e2")
add_node2_e2 = make_node("Add", inputs=["out_add1_e2","b_i"], outputs=["i_t_ba"],name="add_node2_e2")
quant_linear1_e2 = make_node("QuantizeLinear", inputs=["i_t_ba","scale_1","zero_point_all"], outputs=["i_t_ql1"],name="quant_linear1_e2")
dequant_linear1_e2 = make_node("DequantizeLinear", inputs=["i_t_ql1","scale_1", "zero_point_all"], outputs=["i_t_dql1"], name="dequant_linear1_e2")
# id_node_3_e2      = make_node("Identity", inputs=["i_t_dql1"], outputs=["input_acc"], name="id_node_3_e2")
sig_i_e2     = make_node("Sigmoid", inputs=["i_t_dql1"], outputs=["i_t"],name="sig_i_e2")
quant_linear2_e2 = make_node("QuantizeLinear", inputs=["i_t","scale_2","zero_point_unsigned"], outputs=["i_t_ql2"],name="quant_linear2_e2")
dequant_linear2_e2 = make_node("DequantizeLinear", inputs=["i_t_ql2", "scale_2", "zero_point_unsigned"], outputs=["i_t_dql2"], name="dequant_linear2_e2")
# id_node_4_e2      = make_node("Identity", inputs=["i_t_dql2"], outputs=["i_t_gate"], name="id_node_4_e2")

#3rd Equation : Output gate
mul_node1_e3 = make_node("MatMul", inputs=["dql_wo_out","dql_input_out"], outputs=["out_m1_e3"], name="mul_node1_e3")
# id_node_1_e3 = make_node("Identity", inputs=["out_m1_e3"], outputs=["io"], name="id_node_1_e3")
mul_node2_e3 = make_node("MatMul", inputs=["dql_uo_out","h_t-1"], outputs=["out_m2_e3"],name="mul_node2_e3")
# id_node_2_e3 = make_node("Identity", inputs=["out_m2_e3"], outputs=["ho"], name="id_node_2_e3")
add_node1_e3 = make_node("Add", inputs=["out_m1_e3","out_m2_e3"], outputs=["out_add1_e3"],name="add_node1_e3")
add_node2_e3 = make_node("Add", inputs=["out_add1_e3","b_o"], outputs=["o_t_ba"],name="add_node2_e3" )
quant_linear1_e3 = make_node("QuantizeLinear", inputs=["o_t_ba","scale_7","zero_point_all"], outputs=["o_t_ql1"],name="quant_linear_e3")
dequant_linear1_e3 = make_node("DequantizeLinear", inputs=["o_t_ql1","scale_7", "zero_point_all"], outputs=["o_t_dql1"], name="dequant_linear_e3")
# id_node_3_e3      = make_node("Identity", inputs=["o_t_dql1"], outputs=["out_acc"], name="id_node_3_e3")
sig_o_e3     = make_node("Sigmoid", inputs=["o_t_dql1"], outputs=["o_t"],name="sig_o_e3")
quant_linear2_e3 = make_node("QuantizeLinear", inputs=["o_t","scale_8","zero_point_unsigned"], outputs=["o_t_ql2"],name="quant_linear2_e3")
dequant_linear2_e3 = make_node("DequantizeLinear", inputs=["o_t_ql2", "scale_8", "zero_point_unsigned"], outputs=["o_t_dql2"], name="dequant_linear2_e3")
# id_node_4_e3      = make_node("Identity", inputs=["o_t_dql2"], outputs=["o_t_gate"], name="id_node_4_e3")

#4th Equation : Cell gate
mul_node1_e4 = make_node("MatMul", inputs=["dql_wc_out","dql_input_out"], outputs=["out_m1_e4"], name="mul_node1_e4")
# id_node_1_e4 = make_node("Identity", inputs=["out_m1_e4"], outputs=["ic"], name="id_node_1_e4")
mul_node2_e4 = make_node("MatMul", inputs=["dql_uc_out","h_t-1"], outputs=["out_m2_e4"],name="mul_node2_e4")
# id_node_2_e4 = make_node("Identity", inputs=["out_m2_e4"], outputs=["hc"], name="id_node_2_e4")
add_node1_e4 = make_node("Add", inputs=["out_m1_e4","out_m2_e4"], outputs=["out_add1_e4"],name="add_node1_e4")
add_node2_e4 = make_node("Add", inputs=["out_add1_e4","b_c"], outputs=["c_t_ba"],name="add_node2_e4")
quant_linear1_e4 = make_node("QuantizeLinear", inputs=["c_t_ba","scale_5","zero_point_all"], outputs=["c_t_ql1"],name="quant_linear1_e4")
dequant_linear1_e4 = make_node("DequantizeLinear", inputs=["c_t_ql1","scale_5", "zero_point_all"], outputs=["c_t_dql1"], name="dequant_linear1_e4")
# id_node_3_e4      = make_node("Identity", inputs=["c_t_dql1"], outputs=["cell_acc"], name="id_node_3_e4")
tanh_c_e4    = make_node("Tanh", inputs=["c_t_dql1"], outputs=["c_t_partial"],name="tanh_c_e4")
quant_linear2_e4 = make_node("QuantizeLinear", inputs=["c_t_partial","scale_6","zero_point_all"], outputs=["c_t_ql2"],name="quant_linear2_e4")
dequant_linear2_e4 = make_node("DequantizeLinear", inputs=["c_t_ql2", "scale_6", "zero_point_all"], outputs=["c_t_dql2"], name="dequant_linear2_e4")
# id_node_4_e4      = make_node("Identity", inputs=["c_t_dql2"], outputs=["c_t_gate"], name="id_node_4_e4")

#5th Equation : Cell state compute
#id_node_check = make_node("Identity", inputs=["c_t-1"], outputs=["check_c_t"], name="id_node_check")
el_mul_node1_e5 = make_node("Mul", inputs=["f_t_dql2","c_t-1"], outputs=["out_el_mul1_e5"],name="el_mul_node1_e5")
quant_linear1_e5 = make_node("QuantizeLinear", inputs=["out_el_mul1_e5","scale_9","zero_point_all"], outputs=["fifth_ql1"],name="quant_linear1_e5")
dequant_linear1_e5 = make_node("DequantizeLinear", inputs=["fifth_ql1","scale_9", "zero_point_all"], outputs=["fifth_dql1"], name="dequant_linear1_e5")
el_mul_node2_e5 = make_node("Mul", inputs=["i_t_dql2","c_t_dql2"], outputs=["out_el_mul2_e5"], name="el_mul_node2_e5") 
quant_linear2_e5 = make_node("QuantizeLinear", inputs=["out_el_mul2_e5","scale_9","zero_point_all"], outputs=["fifth_ql2"],name="quant_linear2_e5")
dequant_linear2_e5 = make_node("DequantizeLinear", inputs=["fifth_ql2","scale_9", "zero_point_all"], outputs=["fifth_dql2"], name="dequant_linear2_e5")
out_add1_e5     = make_node("Add", inputs=["fifth_dql1","fifth_dql2"], outputs=["c_t"], name="out_add1_e5")
quant_linear3_e5 = make_node("QuantizeLinear", inputs=["c_t","scale_9","zero_point_all"], outputs=["h_t_ql"], name="quant_linear3_e5")
dequant_linear3_e5 = make_node("DequantizeLinear", inputs=["h_t_ql","scale_9","zero_point_all"], outputs=["h_t_dql"], name="dequant_linear3_e5")

#6th Equation : Hidden state compute
tanh_node_e6    = make_node("Tanh", inputs=["h_t_dql"], outputs=["out_tanh_e6"], name="tanh_node_e6") 
quant_linear1_e6 = make_node("QuantizeLinear", inputs=["out_tanh_e6","scale_10","zero_point_all"], outputs=["sixth_ql1"], name="quant_linear1_e6")
dequant_linear1_e6 = make_node("DequantizeLinear", inputs=["sixth_ql1","scale_10","zero_point_all"], outputs=["sixth_dql1"], name="dequant_linear1_e6")
el_mul_node1_e6 = make_node("Mul", inputs=["sixth_dql1","o_t_dql2"], outputs=["h_t_inter"], name="el_mul_node1_e6")#h_t_inter
quant_linear2_e6 = make_node("QuantizeLinear", inputs=["h_t_inter","scale_11","zero_point_all"], outputs=["sixth_ql2"], name="quant_linear2_e6")
dequant_linear2_e6 = make_node("DequantizeLinear", inputs=["sixth_ql2","scale_11","zero_point_all"], outputs=["h_t"], name="dequant_linear2_e6")
id_node_e6      = make_node("Identity", inputs=["h_t"], outputs=["h_t_concat"], name="id_node_e6")

qcdq_lstm_full_quantization = onnx.load("./quant_lstm_full_quantization_qcdq.onnx")
weights = qcdq_lstm_full_quantization.graph.initializer
print(len(weights))
for i in range(len(weights)):
    w = numpy_helper.to_array(weights[i])
    print (qcdq_lstm_full_quantization.graph.initializer[i].name)
    print(w.shape)
    print(w,',',i)
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
        #    id_node_0_e0,
           dql_input, 
        #    id_node_1_e0, 
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
        #    id_node_1_e1,
           mul_node2_e1,
        #    id_node_2_e1, 
           add_node1_e1, 
           add_node2_e1,
           quant_linear1_e1,
           dequant_linear1_e1,
        #    id_node_3_e1,
           sig_f_e1,
           quant_linear2_e1, 
           dequant_linear2_e1,
        #    id_node_4_e1, 
           mul_node1_e2, 
        #    id_node_1_e2,
           mul_node2_e2,
        #    id_node_2_e2, 
           add_node1_e2, 
           add_node2_e2,
           quant_linear1_e2,
           dequant_linear1_e2,
        #    id_node_3_e2,
           sig_i_e2,
           quant_linear2_e2,
           dequant_linear2_e2,
        #    id_node_4_e2, 
           mul_node1_e3, 
        #    id_node_1_e3,
           mul_node2_e3,
        #    id_node_2_e3, 
           add_node1_e3, 
           add_node2_e3,
           quant_linear1_e3,
           dequant_linear1_e3,
        #    id_node_3_e3, 
           sig_o_e3,
           quant_linear2_e3,
           dequant_linear2_e3,
        #    id_node_4_e3, 
           mul_node1_e4, 
        #    id_node_1_e4,
           mul_node2_e4,
        #    id_node_2_e4, 
           add_node1_e4, 
           add_node2_e4,
           quant_linear1_e4,
           dequant_linear1_e4,
           tanh_c_e4,
        #    id_node_3_e4, 
           quant_linear2_e4,
           dequant_linear2_e4,
        #    id_node_4_e4, 
           #id_node_check, 
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
          ],
    name = "QCDQ-LSTM-SCAN",
    inputs=[inp2_m2,inp2_elm1,inp2_m1], #The order in which the inputs are defined here should match the input order when the scan node is defined.
    outputs = [out_hidden_state, out_cell_state, out_hidden_state_concat],
    value_info=[
           # make_tensor_value_info("check_c_t",onnx.TensorProto.FLOAT, [20,1]),
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
            make_tensor_value_info("clp_uo",onnx.TensorProto.INT8, [20,20]),
        ],
    initializer=[make_tensor('W_f',onnx.TensorProto.FLOAT, [20,10], (Wf_val)),
                 make_tensor('U_f',onnx.TensorProto.FLOAT, [20,20], (Uf_val)),
                 make_tensor('b_f',onnx.TensorProto.FLOAT, [20,1], (bf_val)),
                 #Scalars 'scale' and 'zero_point' should be defined as below. Converting them into numpy array based single values causes some errors and exceptions saying that these values should be scalar. The definition has to be like this.
                 # Scalars are tensors with undefined shapes.
               #   make_tensor('scale_all',onnx.TensorProto.FLOAT,[],[1]),
                 make_tensor('inp_scale',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_i',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[15]))]),
                 make_tensor('scale_c',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[18]))]),
                 make_tensor('scale_o',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[19]))]),
                 make_tensor('scale_f',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[20]))]),
                 make_tensor('scale_1',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]), #0.0057...
                 make_tensor('scale_2',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]), #0034227842
                 make_tensor('scale_3',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
               #   make_tensor('scale_test',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),#Correct input scale : 0.00781916
                 make_tensor('scale_4',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_5',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_6',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_7',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[22]))]), #Approximate scale_7 value = 0.0085895785
                 make_tensor('scale_8',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[22]))]),#0.0026683041
                 make_tensor('scale_9',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),
                 make_tensor('scale_10',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[22]))]),
                 make_tensor('scale_11',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[12]))]),#0.0036052174                 
                 make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 make_tensor('zero_point_unsigned',onnx.TensorProto.UINT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 #Introducing scalars for the clip operators.
                 make_tensor('min', onnx.TensorProto.INT8, [], [-127]),
                 make_tensor('max', onnx.TensorProto.INT8, [], [127]),
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
onnx.save(onnx_model, './quantize-lstm-full-graph-test.onnx')

#Converting to opset version '14' to accomodate clip nodes with INT8 and UINT8 input 
onnx_model.opset_import[0].version = 14
# print(onnx_model)

# Testing to check if the model is serializing without errors or warnings
#Even after converting to an opset version of 14 there was an error saying that the clip operator is tied to two different datatypes (int8 and float)
#That was because the MIN and the MAX values were defined as FLOAT tensors and the Clip operator constrains the input and output datatypes to be the same.
#Converting them to INT8 datatypes solved that error.
sess = rt.InferenceSession(onnx_model.SerializeToString())

# Defining the values of the varibales to test the execution of the onnx model
# in1lstm =  np.ones((10, 1)).astype(np.float32)
# print(in1lstm)
in1lstm = np.empty([10,1],dtype=np.float32).reshape([10,1])
#Input values have to be below 0. Values greater than 1 will be quantized with the same values in the quantizerlinear layer and there will be no chnage in the outputs.
#For vallues between 0 and 1 we get similar outputs from the brevitas layer and this graph. Clean and organize this code properly tomorrow.
in1lstm.fill(0.5)
# print(in1lstm)
in2lstm =  np.zeros((20, 1)).astype(np.float32)
in3lstm =  np.zeros((20, 1)).astype(np.float32)
input_dict = {}
input_dict["X"] = in1lstm
input_dict["h_t-1"] = in2lstm
input_dict["c_t-1"] = in3lstm 
# print(input_dict)

#Executing the onnx model here.
sess = rt.InferenceSession(onnx_model.SerializeToString())
output = sess.run(None, input_dict)
print(output)
print('------------------------------')
exit()
# print('Forget gate = ',output[3].reshape([1,20]))
# print('Input gate = ',output[4].reshape([1,20]))
# print('Cell gate = ',output[5].reshape([1,20]))
# print('Output gate = ',output[6].reshape([1,20]))
# print('h_f_matmul = ',output[7].reshape([1,20]))
# print('i_f_matmul = ',output[8].reshape([1,20]))
# print('Input after quantization = ',output[3].reshape([1,10]))
# print('------------------------------')

hidden_out = np.load('./hidden.npy')

# quant_ii = np.load('./quant_ii_gate.npy')
# quant_hi = np.load('./quant_hi_gate.npy')
# quant_input_acc = np.load('./quant_input_acc_gate.npy')
# input_out = np.load('./quant_input_gate.npy')

# quant_if = np.load('./quant_if_gate.npy')
# quant_hf = np.load('./quant_hf_gate.npy')
# quant_forget_acc = np.load('./quant_forget_acc_gate.npy')
# forget_out = np.load('./quant_forget_gate.npy')

# quant_ic = np.load('./quant_ic_gate.npy')
# quant_hc = np.load('./quant_hc_gate.npy')
# quant_cell_acc = np.load('./quant_cell_acc_gate.npy')
# cell_out = np.load('./quant_cell_gate.npy')

# quant_io = np.load('./quant_io_gate.npy')
# quant_ho = np.load('./quant_ho_gate.npy')
# quant_out_acc = np.load('./quant_out_acc_gate.npy')
# out_out = np.load('./quant_out_gate.npy')

##Outputs from my graph

my_hidden_out = output[0].reshape([1,20])

# my_ii_out = output[4].reshape([1,20])
# my_hi_out = output[5].reshape([1,20])
# my_input_gate_acc = output[6].reshape([1,20])
# my_input_out = output[7].reshape([1,20])

# my_if_out = output[8].reshape([1,20])
# my_hf_out = output[9].reshape([1,20])
# my_forget_gate_acc = output[10].reshape([1,20])
# my_forget_out = output[11].reshape([1,20])

# my_ic_out = output[12].reshape([1,20])
# my_hc_out = output[13].reshape([1,20])
# my_cell_gate_acc = output[14].reshape([1,20])
# my_cell_out = output[15].reshape([1,20])

# my_io_out = output[16].reshape([1,20])
# my_ho_out = output[17].reshape([1,20])
# my_out_gate_acc = output[18].reshape([1,20])
# my_out_out = output[19].reshape([1,20])

print('My_Hidden out = ',my_hidden_out)
print('Brevitas_Hidden_out = ',hidden_out[0])
print('------------------------------')

scale = 0.001#0.005

# scale_ii = 0.0057
# scale_hi = 0.001760039
# scale_if = 0.0057
# scale_hf = 0.001760039
# scale_ic = 0.0057
# scale_hc = 0.001760039
# scale_io = 0.0057
# scale_ho = 0.001760039
# scale_1 = 0.00781916
# scale_2 = 0.0034227842
# scale_3 = 0.006515566
# scale_4 = 0.00731916
# scale_5 = 0.005926438
# scale_6 = 0.0071433834
# scale_7 = 0.0085895785
# scale_8 = 0.0026683041
# scale_hidden  = 0.0036052174

scale_ii = 0.0078125
scale_hi = 0.00175792
scale_if = 0.0057
scale_hf = 0.001760039
scale_ic = 0.0057
scale_hc = 0.001760039
scale_io = 0.0057
scale_ho = 0.001760039

W_i = 0.0017548618
U_i = 0.0017579200211912394
W_f = 0.0017538558
U_f = 0.0017540532862767577
W_c = 0.0017536403611302376
U_c = 0.0017550183
W_o = 0.0017583035
U_o = 0.0017599976854398847
scale_1 = 0.0078125
scale_2 = 0.0078125
scale_3 = 0.0078125
scale_4 = 0.0078125
scale_5 = 0.0078125
scale_6 = 0.0078125
scale_7 = 0.003921568859368563
scale_8 = 0.003921568859368563
scale_9 = 0.0078125
scale_10 = 0.003921568859368563
scale_11 = 0.0078125
scale_hidden = 0.0078125

# diff_ii = np.round((my_ii_out - quant_ii)/scale_ii)
# diff_hi = np.round((my_hi_out - quant_hi)/scale_hi)
# diff_input_acc = np.round((my_input_gate_acc - quant_input_acc)/scale_1) 
# diff_input = np.round((my_input_out - input_out)/scale_2)
# diff_if = np.round((my_if_out - quant_if)/scale_ii)
# diff_hf = np.round((my_hf_out - quant_hf)/scale_hi)
# diff_forget_acc =  np.round((my_forget_gate_acc - quant_forget_acc)/scale_3)
# diff_forget = np.round((my_forget_out - forget_out)/scale_4)
# diff_ic = np.round((my_ic_out - quant_ic)/scale_ii)
# diff_hc = np.round((my_hc_out - quant_hc)/scale_hi)
# diff_cell_acc = np.round((my_cell_gate_acc - quant_cell_acc)/scale_5)
# diff_cell = np.round((my_cell_out - cell_out)/scale_6)
# diff_io = np.round((my_io_out - quant_io)/scale_ii)
# diff_ho = np.round((my_ho_out - quant_ho)/scale_hi)
# diff_out_acc = np.round((my_out_gate_acc - quant_out_acc)/scale_7)
# diff_out = np.round((my_out_out - out_out)/scale_8)

diff_hidden = np.round((my_hidden_out - hidden_out[0])/scale_hidden)

# print('Diff ii = ',diff_ii)
# print('Diff hi = ',diff_hi)
# print('DIff input acc = ', diff_input_acc)
# print('DIff input gate = ',diff_input)

# print('Diff if = ',diff_if)
# print('Diff hf = ',diff_hf)
# print('DIff forget acc = ', diff_forget_acc)
# print('DIff forget gate = ',diff_forget)

# print('Diff ic = ',diff_ic)
# print('Diff hc = ',diff_hc)
# print('DIff cell acc = ', diff_cell_acc)
# print('DIff cell gate = ',diff_cell)

# print('Diff io = ',diff_io)
# print('Diff ho = ',diff_ho)
# print('DIff output acc = ', diff_out_acc)
# print('DIff output gate = ',diff_out)

# print('First input out = ',output[20].reshape([1,10]))

print('Diff hidden = ', diff_hidden)

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
print("Scan Node graph definition is here")
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
onnx.save(lstm_scan_node_model_14, './lstm_scan_node_model_14.onnx')
# Defining the values of the varibales to test the execution of the onnx model
in1_inpa =  np.zeros((20, 1)).astype(np.float32)#'h_t-1'
in2_inpb = np.zeros((20, 1)).astype(np.float32)#'c_t-1'
# in3_scan_input =  np.ones((5, 10, 1)).astype(np.float32)#'X' 10,1 : Because that is the way the shape of the model has been defined.
in3_scan_input = np.empty([25,10,1],dtype=np.float32).reshape([25,10,1])
in3_scan_input.fill(0.5)
input_dict = {}
input_dict["inp_a"] = in1_inpa
input_dict["inp_b"] = in2_inpb
input_dict["scan_input"] = in3_scan_input

#Executing the onnx model here.
sess = rt.InferenceSession(lstm_scan_node_model_14.SerializeToString())
output = sess.run(None, input_dict)
# print("Final Hidden State = ", output[0].reshape([1,20]))
# print("Final Cell State = ", output[1].reshape([1,20]))
# print("All Hidden States = ", output[2].reshape([5,1,20]))

brevitas_out = np.load('./hidden.npy')
# print(brevitas_out[4])
scan_full_out = output[2].reshape([25,1,20])
# print(scan_full_out[4])
comp = scan_full_out - brevitas_out
# print(comp)
scale_hidden_128 = 0.0078125
scale_hidden_256 = 0.003921569
round_comp = np.round(comp/scale_hidden_128)
print(round_comp)