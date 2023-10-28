import torch
from brevitas.nn import QuantLSTM
from brevitas.export import export_onnx_qcdq
import numpy as np
import torch.nn as nn

torch.manual_seed(0)

# quant_lstm_weight_only = QuantLSTM(input_size=10, hidden_size=20, weight_bit_width=4, io_quant=None, bias_quant=None, gate_acc_quant=None, sigmoid_quant=None, tanh_quant=None, cell_state_quant=None)
model_lstm = nn.Sequential( 
    QuantLSTM(input_size=10, hidden_size=20,bias_quant=None,weight_quant=None)
    )

export_path = 'quant_lstm_no_bias_no_weight_quantization_qcdq.onnx'
# quant_lstm_full_quantization =  QuantLSTM(input_size=10, hidden_size=20,bias_quant=None,weight_quant=None)
export_onnx_qcdq(model_lstm, (torch.randn(25, 1, 10)), opset_version=14, export_path=export_path)
model_lstm.eval()
# in_qcdq_node =  np.ones((5,1,10)).astype(np.float32)#Okay, so when I give this shape, the values are not repeated. 
in_qcdq_node = np.empty([25,1,10],dtype=np.float32).reshape([25,1,10])
in_qcdq_node.fill(1)
input_test = torch.from_numpy(in_qcdq_node)
output_lstm = model_lstm(input_test)
# print(output_lstm)
output_lstm = output_lstm[0].detach().numpy()#.detach().numpy()
print(type(output_lstm))
print(output_lstm)
np.save('hidden_no_bw.npy',output_lstm)