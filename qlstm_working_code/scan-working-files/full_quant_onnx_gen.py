import torch
from brevitas.nn import QuantLSTM
from brevitas.export import export_onnx_qcdq
import numpy as np
import torch.nn as nn

torch.manual_seed(0)

model_lstm = nn.Sequential( 
    QuantLSTM(input_size=10, hidden_size=20,bias_quant=None)
    )#, cache_inference_quant_bias=True
model_lstm.eval()
export_path = 'quant_lstm_full_quantization_qcdq.onnx'
export_onnx_qcdq(model_lstm,(torch.randn(25, 1, 10)), opset_version=14, export_path=export_path)#(torch.randn(25, 1, 10))
in_qcdq_node = np.empty([25,1,10],dtype=np.float32).reshape([25,1,10])
in_qcdq_node.fill(0.5)
print('Supplied Input = ',in_qcdq_node[0][0][0])
input_test = torch.from_numpy(in_qcdq_node)
output_lstm = model_lstm(input_test)
output_lstm = output_lstm[0].detach().numpy()#.detach().numpy()
print(type(output_lstm))
print(output_lstm)
np.save('hidden.npy',output_lstm)