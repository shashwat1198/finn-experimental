import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

#Defining the input activation bit width below and from that generating all possible INT8 inputs in that range.
acivation_bit_width = 8
num_thresholds = 2 ** acivation_bit_width - 1
# int_input_range = np.arange(start= - 2 ** (acivation_bit_width - 1),
#                                     stop= 2 ** (acivation_bit_width - 1), #stop is not included, so last value is stop -1
#                                     dtype=np.float32)
int_input_range = np.linspace(-0.501, 0.498, num=255) #Num of inputs is 255 as thersholds are the number of inputs which cause a change in the output integer levels. Since num_thresholds == 255 for 8-bit activation hence number of inputs is 255
print('Input Range based on the input activation bit width')
print(int_input_range)
int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
tanh_out = nn.Tanh()              #defining sigmoid activation
output = tanh_out(int_input_range)   #output of the activation function
print('Tanh_out')
print(output) 

#QuantizeLinear operation
scale = 0.003921568859368563
zero_point = 128 #This was different from the sigmoid activation.
output = torch.round(output/scale)
#Factoring in the zero point
output = output + zero_point 
print('Scaled & shifted output of the quantization of the tanh outputs')
print(output)

#Need to clip the outputs here to repliacte the fucnctioning of the Quant Node.
#Clamping operation
# min_int_val = -128
# max_int_val = 127
# output = np.where(output > max_int_val,max_int_val, output)
# output = np.where(output < min_int_val,min_int_val, output)

print('Output of the Clip operation in range [-128,127]')
print(output)

unique_input_indices = np.unique(output, return_index=True)[1] #[1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
print('Unique input indices')
print(unique_input_indices)

unique_inputs = np.zeros(len(unique_input_indices))
unique_inputs = int_input_range[unique_input_indices]#[1:] #identifying these inputs from the input_range array and ignoring the first threshold
print('Unique Inputs')
print(unique_inputs)
print('First uniqe input')
print(unique_inputs[0])

print('-------- Hopefully these are the thersholds -------------------------')
thresholds = np.zeros(num_thresholds)
# Inserting logic to incorporate repeat values in the thresholds. Here I am working directly with the values of the matrix. Alessandro has first isolated the indices then did the repeats then using these indices used to identify the inputs which changed the outputs.
threshold_index = 0
output_index = 0
index_pos = 0
while threshold_index < 255 and output_index < 254 and index_pos < len(unique_input_indices):
    if output[output_index] == output[output_index+1]: 
        output_index += 1
    elif output[output_index] != output[output_index+1]:
        if(index_pos == 0):
            diff = output[0]
        elif(index_pos != 0):
            diff = output[output_index+1] - output[output_index] # Calculating number of required repeats
        while diff > 0:  #Copying repeats into the threshold thershold matrix       
            #print("Index pos",index_pos)
            #thresholds[threshold_index] = int_input_range[unique_input_indices[index_pos]] # 
            thresholds[threshold_index] = unique_inputs[index_pos] 
            threshold_index += 1
            diff -= 1
        output_index += 1
        index_pos += 1


print('Threshold Index = ',threshold_index)
for i in range(threshold_index,255):
    #print('Unique Input Index = ',unique_input_indices[index_pos-1])
    thresholds[i] = (int_input_range[unique_input_indices[index_pos]])
    
print('Modified Thresholds with the right amount of repeats')
print(thresholds)

# To buid the multi-thresold matrix, two matrices are required:
#1. An array with the inputs that change the integer level in the output. These are the inputs that go in the final threshold array.
#2. The quantized output matrix of the given activation function with the scale and the zero-point applied to this matrix. This matrix will give information on how many times each threshold needs to be repeated in the final threshold array 

num_inputs = 255
out_array = np.zeros(num_inputs)

#MT comparison operation.
print('MT output')
for i in range(num_inputs):
    start_index = 0        #For each input scanning the input threshold matrix to find the possible position of the threhsold
    while thresholds[start_index] <= int_input_range[i]: #and start_index < 254
        start_index += 1
        if start_index == 254:
            start_index = 255
            break
    out_array[i] =  start_index #np.minimum(start_index,127) For INT8 QuantLinear; clipping the output at 127

print('Multithreshold op. output')
#Not sure if the zero_point will be included like this in the operation. Have to check this with this the team
print(out_array)
