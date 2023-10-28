import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

#Defining the input activation bit width below and from that generating all possible INT8 inputs in that range.
acivation_bit_width = 8
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

##Identity unique indices from all possible floating point values.
values, indices = np.unique(output, return_index=True)
indices = np.unique(output, return_index=True)[1]
print('Unique Values')
print(values)
print('Unique Indices : Inputs which cause a change in the output values of the activation')
print(indices)
print('No. of indices = ', len(indices))

num_thresholds = 255
unique_index_matrix = np.zeros(255)
modified_index = 0
threshold_index = 0
unique_index_matrix[modified_index] = indices[threshold_index]
modified_index += 1
threshold_index += 1
print(len(indices))
while modified_index < 255 and threshold_index < len(indices):
    #print(threshold_index)
    #print(indices[threshold_index+1])
    if threshold_index != len(indices) - 1:
        if indices[threshold_index] != indices[threshold_index+1]:
            diff = indices[threshold_index+1] - indices[threshold_index] # Calculating number of required repeats
            while diff > 0:  #Copying repeats into the modified thershold matrix       
                unique_index_matrix[modified_index] = indices[threshold_index] #Discarding the first threshold value from the matrix with the '+1' value
                modified_index += 1
                diff -= 1
            threshold_index += 1
    if threshold_index == len(indices) - 1: #if we are at the last element on the unique index array
        diff = num_thresholds - modified_index
        while diff > 0:  #Copying repeats into the modified thershold matrix       
            unique_index_matrix[modified_index] = indices[threshold_index] #Discarding the first threshold value from the matrix with the '+1' value
            modified_index += 1
            diff -= 1
        threshold_index += 1

print('Modified Index = ',modified_index)
print('Threshold Index = ',threshold_index)

# for i in range(modified_index,255):
#     unique_index_matrix[i] = unique_index_matrix[modified_index-1]

print('Unique Index Matrix')
print(unique_index_matrix)

index_threshold_matrix = int_input_range[unique_index_matrix]
print('Index threshold matrix')
print(index_threshold_matrix)

#These are the Quantize linear operations
#Mul with the scale factor
scale_out = torch.round(output*255)
#Factoring in the zero point
scale_out = scale_out - 128 
print('Scaled Output of the Quantize Linear Layer')
print(scale_out)

modified_threshold_matrix = np.zeros(255)

#Inserting logic to incorporate repeat values in the thresholds. Here I am working directly with the values of the matrix. Alessandro has first isolated the indices then did the repeats then using these indices used to identify the inputs which changed the outputs.
# modified_index = 0
# threshold_index = 0
# while modified_index < 255:
#     if scale_out[threshold_index] == scale_out[threshold_index+1]: 
#         threshold_index += 1
#     elif scale_out[threshold_index] != scale_out[threshold_index+1]:
#         diff = scale_out[threshold_index+1] - scale_out[threshold_index] # Calculating number of required repeats
#         while diff > 0:  #Copying repeats into the modified thershold matrix       
#             modified_threshold_matrix[modified_index] = scale_out[threshold_index+1] #Discarding the first threshold value from the matrix with the '+1' value
#             modified_index += 1
#             diff -= 1
#         threshold_index += 1

# print('Modified Threshold Matrix')
# print(modified_threshold_matrix)

print('Modified Threshold Matrix')
print(modified_threshold_matrix)

#Threshold matrix
threshold_matrix = int_input_range[indices]
print('Threshold Matrix')
print(threshold_matrix)

#sigmoid_scale = 0.00392156862
#Division and add with the scale in the quantize linear layer
#div_sigmoid = output/sigmoid_scale
#print('Quantize Linear output')
#print(div_sigmoid)

#Subtract & multiply with the scale to get the float tensor back.
#mul_sigmoid = div_sigmoid * sigmoid_scale
#print('DeQuantize Linear output')
#print(mul_sigmoid)

num_inputs = 20
out_index = np.zeros(num_inputs)
in_X = np.array([-128,67,89,25,127,0,2,4,12,-36, -65, -29, 48, 1, 91, 126, 21, 97, -15, 56],dtype=np.float32)

#MT comparison operation.
print('MT output')
for i in range(num_inputs):
    start_index = 0        #For each input scanning the input threshold matrix to find the possible position of the threhsold
    while in_X[i] > index_threshold_matrix[start_index] and start_index < 254:
        start_index += 1
    out_index[i] = np.minimum(start_index,127) #For INT8 QuantLinear; clipping the output at 127
    print(out_index[i]) 

# 0.0
# 127.0
# 127.0
# 127.0
# 127.0
# 89.0
# 91.0
# 93.0
# 101.0
# 53.0
# 24.0
# 60.0
# 127.0
# 90.0
# 127.0
# 127.0
# 127.0
# 127.0
# 74.0
# 127.0

# QuantLinear output
# [[  0]
#  [127]
#  [127]
#  [127]
#  [127]
#  [ 64]
#  [113]
#  [126]
#  [127]
#  [  0]
#  [  0]
#  [  0]
#  [127]
#  [ 94]
#  [127]
#  [127]
#  [127]
#  [127]
#  [  0]
#  [127]]