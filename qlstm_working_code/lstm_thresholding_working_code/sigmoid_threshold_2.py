import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

#Defining the input activation bit width below and from that generating all possible INT8 inputs in that range.
acivation_bit_width = 8
num_thresholds = 2 ** acivation_bit_width - 1
int_input_range = np.arange(start= - 2 ** (acivation_bit_width - 1),
                                    stop= 2 ** (acivation_bit_width - 1), #stop is not included, so last value is stop -1
                                    dtype=np.float32)
print('Input Range based on the input activation bit width')
print(int_input_range)
int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
sigmoid_out = nn.Sigmoid()              #defining sigmoid activation
output = sigmoid_out(int_input_range)   #output of the activation function
print('Sigmoid_out')
print(output) 

#Quantization operation
scale = 0.003921568859368563
zero_point = -128
output = torch.round(output/scale)
#Factoring in the zero point
output = output + zero_point 
print('Scaled & shifted output of the quantization of the sigmoid outputs')
print(output)

unique_input_indices = np.unique(output, return_index=True)[1][1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
print('Unique input indices')
print(unique_input_indices)

thresholds = np.zeros(len(unique_input_indices))
thresholds = int_input_range[unique_input_indices[1:]] #identifying these inputs from the input_range array and ignoring the first threshold
print('Input thresholds')
print(thresholds)

print('-------- Hopefully these are the thersholds -------------------------')
modified_thresholds = np.zeros(num_thresholds)
# Inserting logic to incorporate repeat values in the thresholds. Here I am working directly with the values of the matrix. Alessandro has first isolated the indices then did the repeats then using these indices used to identify the inputs which changed the outputs.
modified_index = 0
threshold_index = 0
index_pos = 0
while modified_index < 255:
    if output[threshold_index] == output[threshold_index+1]: 
        threshold_index += 1
    elif output[threshold_index] != output[threshold_index+1]:
        diff = output[threshold_index+1] - output[threshold_index] # Calculating number of required repeats
        while diff > 0:  #Copying repeats into the modified thershold matrix       
            modified_thresholds[modified_index] = int_input_range[unique_input_indices[index_pos]] 
            modified_index += 1
            diff -= 1
        threshold_index += 1
        index_pos += 1
    
print('Modified Thresholds with the right amount of repeats')
print(modified_thresholds)

# To buid the multi-thresold matrix, two matrices are required:
#1. An array with the inputs that change the integer level in the output. These are the inputs that go in the final threshold array.
#2. The quantized output matrix of the given activation function with the scale and the zero-point applied to this matrix. This matrix will give information on how many times each threshold needs to be repeated in the final threshold array 

num_inputs = 256
out_array = np.zeros(num_inputs)

#MT comparison operation.
print('MT output')
for i in range(num_inputs):
    start_index = 0        #For each input scanning the input threshold matrix to find the possible position of the threhsold
    while modified_thresholds[start_index] <= int_input_range[i]: #and start_index < 254
        start_index += 1
        if start_index == 255:
            break
    out_array[i] =  start_index #np.minimum(start_index,127) For INT8 QuantLinear; clipping the output at 127

print('Multithreshold op. output')
#Not sure if the zero_point will be included like this in the operation. Have to check this with this the team
print(out_array+zero_point)
