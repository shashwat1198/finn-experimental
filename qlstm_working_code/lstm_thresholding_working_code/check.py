import numpy as np
#Shashwat torch imports
import torch
from torch import nn

int_input_range = np.linspace(-0.5, 0.5, num=255) #Generating 255 values between -0.5 and 0.5
print(int_input_range)
int_input_range = torch.from_numpy(int_input_range)              #conversion to torch tensor
            #Sigmoid computation
sigmoid_out = nn.Sigmoid()                                       #defining sigmoid activation
output = sigmoid_out(int_input_range)
print('Sigmoid float Output')
print(output)
#print('Output Length = ',len(output))
#QuantizeLinear operation
zero_point = -128
quant_scale = (1/255)
output = torch.round(output/quant_scale)
output = output + zero_point
print('Sigmoid Scaled Output')
print(output)
#Clamping operation
min_int_val = -127
max_int_val = 127
output = np.where(output > max_int_val,max_int_val, output)
output = np.where(output < min_int_val,min_int_val, output)
            #scale = 0.003921568393707275
print('Clamped output')
print(output)
unique_input_indices = np.unique(output, return_index=True)[1][1:]
print(unique_input_indices)

            #start_val = -1
            #stop_val = 1
            num_thresholds = int(num_distinct_values - 1)
            #int_input_range = np.arange(start= start_val, stop= (2 ** (bit_width - 1) - 1), dtype=np.float32)
            int_input_range = np.linspace(-0.5, 0.5, num=255) #Generating 255 values between -0.5 and 0.5
            print(len(int_input_range))
            int_input_range = torch.from_numpy(int_input_range)              #conversion to torch tensor
            #Sigmoid computation
            sigmoid_out = nn.Sigmoid()                                       #defining sigmoid activation
            output = sigmoid_out(int_input_range)
            #print('Output Length = ',len(output))
            #QuantizeLinear operation
            zero_point = -128
            output = torch.round(output/quant_scale)
            output = output + zero_point
            #Clamping operation
            min_int_val = -127
            max_int_val = 127
            output = np.where(output > max_int_val,max_int_val, output)
            output = np.where(output < min_int_val,min_int_val, output)
            #scale = 0.003921568393707275
            unique_input_indices = np.unique(output, return_index=True)[1][1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
            #unique_inputs = np.zeros(len(unique_input_indices))
            #unique_inputs = int_input_range[unique_input_indices[1:]] #identifying these inputs from the input_range array and ignoring the first threshold
            thresholds = np.zeros(num_thresholds)#num_thresholds
            modified_index = 0
            threshold_index = 0
            index_pos = 0
            while modified_index < 255 and threshold_index < 254:
                if output[threshold_index] == output[threshold_index+1]: 
                    threshold_index += 1
                elif output[threshold_index] != output[threshold_index+1]:
                    diff = output[threshold_index+1] - output[threshold_index] # Calculating number of required repeats
                    while diff > 0:  #Copying repeats into the modified thershold matrix       
                        thresholds[modified_index] = (int_input_range[unique_input_indices[index_pos]])*quant_scale
                        modified_index += 1
                        diff -= 1
                    threshold_index += 1
                    index_pos += 1
            
            #print('Modified Index = ',modified_index)
            #for i in range(modified_index,255):
                #print('Unique Input Index = ',unique_input_indices[index_pos-1])
            #    thresholds[i] = (int_input_range[unique_input_indices[index_pos-1]])