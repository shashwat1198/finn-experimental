import torch
from brevitas.nn import QuantLSTM
from brevitas.export import export_onnx_qcdq
import numpy as np
import torch.nn as nn

# torch.manual_seed(0)

model_lstm = nn.Sequential(
    QuantLSTM(input_size=10, hidden_size=20, weight_bit_width=4, io_quant=None, bias_quant=None, gate_acc_quant=None, sigmoid_quant=None, tanh_quant=None, cell_state_quant=None)
)
model_lstm.eval()
export_path = 'quant_lstm_weight_only_4b_25.onnx'
export_onnx_qcdq(model_lstm, (torch.randn(25, 1, 10)), opset_version=14, export_path=export_path)

# in_qcdq_node =  np.ones((5,10)).astype(np.float32)
# input_test = torch.from_numpy(in_qcdq_node)
# output_lstm = quant_lstm_weight_only(input_test)
# print(output_lstm)
