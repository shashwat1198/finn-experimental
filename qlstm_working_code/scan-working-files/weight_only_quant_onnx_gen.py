import torch
from brevitas.nn import QuantLSTM
from brevitas.export import export_onnx_qcdq
import numpy as np
import torch.nn as nn

#Setting manual seeds for same execution.
#torch.manual_seed(0)

#Defining a single QuantLSTM layer with an input size of 10 and hidden size of 20 (no. of hidden units)
model_lstm = nn.Sequential(
    QuantLSTM(input_size=10, hidden_size=20, weight_bit_width=4, io_quant=None, bias_quant=None, gate_acc_quant=None, sigmoid_quant=None, tanh_quant=None, cell_state_quant=None)
)
#Need the model.eval() statement before exporting the graph to make sure the learned parameters do not change during execution.
model_lstm.eval()
export_path = 'quant_lstm_weight_only_4b_25.onnx'
export_onnx_qcdq(model_lstm, (torch.randn(25, 1, 10)), opset_version=14, export_path=export_path)

#Testing the brevitas model here.
in_qcdq_node =  np.ones((5,10)).astype(np.float32)
input_test = torch.from_numpy(in_qcdq_node)
output_lstm = model_lstm(input_test)
print(output_lstm)
