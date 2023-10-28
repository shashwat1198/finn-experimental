import torch
from brevitas.nn import QuantLSTM
from brevitas.export import export_onnx_qcdq
import numpy as np
import torch.nn as nn

torch.manual_seed(0)

# quant_lstm_weight_only = QuantLSTM(input_size=10, hidden_size=20, weight_bit_width=4, io_quant=None, bias_quant=None, gate_acc_quant=None, sigmoid_quant=None, tanh_quant=None, cell_state_quant=None)
model_lstm = nn.Sequential( 
    QuantLSTM(input_size=10, hidden_size=20,bias_quant=None)
    )#, cache_inference_quant_bias=True
model_lstm.eval()
num_inputs = 5
num_features = 10
export_path = 'quant_lstm_full_quantization_qcdq_check.onnx'
export_onnx_qcdq(model_lstm,(torch.randn(num_inputs, 1, num_features)), opset_version=14, export_path=export_path)#(torch.randn(25, 1, 10))
# in_qcdq_node =  np.ones((5,1,10)).astype(np.float32)#Okay, so when I give this shape, the values are not repeated. 
in_qcdq_node = np.empty([num_inputs,1,num_features],dtype=np.float32).reshape([num_inputs,1,num_features])
in_qcdq_node.fill(0.5)
print('Supplied Input = ',in_qcdq_node[0][0][0])
input_test = torch.from_numpy(in_qcdq_node)
output_lstm = model_lstm(input_test)
# print(output_lstm)
output_lstm = output_lstm[0].detach().numpy()#.detach().numpy()
print(type(output_lstm))
print(output_lstm)
np.save('hidden.npy',output_lstm)