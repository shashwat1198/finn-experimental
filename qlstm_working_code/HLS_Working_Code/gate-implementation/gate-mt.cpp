//This is a single gate in an LSTM layer that is implemented in this HLS design.
//The design is a 'Static' variation where each input in a sequence is processed after the completion of the current input processing (II = Latency of a single input processing).
//The design can be made 'Non-Static' (utilizing parallelism across sequences) by instantiing multiple such blocks together. (II = 1 can be achieved.)

#include "gate-mt-header.h"
#include <stdio.h>
#include <algorithm>

	// These are the static variables which will be initialized in the start and will act as memory.
	// The arrays get inloop_mv1_2itialized with the specified values at the start of the execution.
	// Each time a function is executed the array remembers it's value from the previous execution.
	// The variables with the 'static' qualifier are also initialized in the RTL design and the FPGA bitstream, removing the need for multiple clock cycles,
	// to initialize the memory and ensures initializing large memories is not an operational overhead.
	// If the 'static' qualifier is not used---------
	// Each time a function with this array is executed, the RAM that implements the array is loaded with these values. For a single-port RAM it takes 'n',
	// clock cycles to load an array of length 'n'.
	// Static variables are registers in the final RTL design.

	//All the activation calculation part should be in the test_bench and the final computed values can then be passed to the LSTM compute function in the form of a 1-D array.
	// Multiple such LUT based activations can be passed here! So no need to compute them actually on the hardware.

	// ARRAY_PARTITION : This pragma allows the arrays to be partitioned into smaller parts, which can then be stored in different memory banks.
	// This can help improve memory bandwidth and access parallelism by allowing multiple memory banks to be accessed concurrently although at an increase in resource utilization.
	// The pragma works by dividing the array across the specified dimension into multiple arrays so that multiple data points are available in a single clock cycle!

	// DATAFLOW : Enables task level pipelining, allowing functions and loops to overlap in their operation, increasing overall throughput of the design.

void max_value(mul_t val,out_t *final_val){
	if(val > 127){
		*final_val = 127;
	}
	else if(val < -128){
		*final_val = -128;
	}
	else{
		*final_val = val;
	}
	//cout << "Final Value = " << *final_val << "\n";
}

void tanh_activation_function(out_t G_out[Out_N], tanh_t tanh_activation[Act_N], out_t G_out_act[Out_N]){
//Once the look up table for the activation function is initialized then we lookup the table array based on the data value.
	int index;
//This loop got pipelined automatically without the need for any directives.
	loop_tanh_activation : for(int b = 0; b < Out_N; b++){
		index = G_out[b];
		if (index < -128)
			index = -128;
		if(index > 127)
			index = 127;
		G_out_act[b] = tanh_activation[index+128];
//+128 because I know my activation response for the positive valus start from the index 128 which is exactly smilar to the sigmiod activation function.
	}
}

//Arrays must be sized even for function arguements. Unsized arrays are not supported.

void gate_function(inp_t X[Inp_N],inp_t H[Out_N], inp_t gate_out[Out_N], param_t weight[Out_N][Inp_N],param_t recurrent[Out_N][Out_N], param_t bias[Out_N], sigmoid_t sigmoid_activation[Act_N]){
	int i,j;
	inp_t mv_1[Out_N];
	inp_t mv_2[Out_N];

	loop_bias_intermediate: for(i=0;i<Out_N;i++){
		mv_1[i] = 0;
		mv_2[i] = 0;
		}
#pragma HLS ARRAY_PARTITION variable = weight dim = 1 complete
//First Matrix-Vector Multiply
	loop_mv1_1:for(i=0;i<Out_N;++i){
		loop_mv1_2:for(j=0;j<Inp_N;++j){
			mv_1[i] = mv_1[i] + weight[i][j]*X[j];
	}
}

//Second Matrix-Vector Multiply
#pragma HLS ARRAY_PARTITION variable = recurrent dim = 1 complete
	loop_mv2_1:for(i=0;i<Out_N;i++){
		loop_mv2_2:for(j=0;j<Out_N;j++)
			mv_2[i] = mv_2[i] + recurrent[i][j]*H[j];
	}

//Addition operation
	loop_add:for(i=0;i<Out_N;i++){
		gate_out[i] = mv_1[i]+mv_2[i]+bias[i];
	}
	sigmoid_activation_function(gate_out,sigmoid_activation);
//	for(i=0;i<1;i++){
//		cout << "Sigmoid loop = " << gate_out[i] << "\n";
//	}
}

void sigmoid_activation_function(out_t G_out[Out_N], sigmoid_t sigmoid_activation[Act_N]){
//Once the look up table for the activation function is initialized then we lookup the table array based on the data value.
	int data_round;
	int index;
	loop_sigmoid_activation : for(int b=0; b<Out_N;b++){
		index = G_out[b];
		if (index < -128) //Updated index for INT8 quantization.
			index = -128;
		if(index > 127)
			index = 127;
		G_out[b] = sigmoid_activation[index+128];
//As the response of the activation to the positive values starts from the index 128.
//INT8 values are from -128 to 127. So a value with index in negative needs to be converted to a positive index by the addition of the zero-point.
	}
}

void gate_function_top(
		inp_t X[Inp_N],inp_t H[Out_N],
		inp_t gate_out[Out_N],
		param_t weight_f[Out_N][Inp_N],
		param_t recurrent_f[Out_N][Out_N],
		param_t bias_f[Out_N],
		sigmoid_t sigmoid_activation[Act_N], tanh_t tanh_activation[Act_N]
){
//Intermediate variables
	gate_function(
			X, H,
			gate_out, weight_f, recurrent_f , bias_f,
			sigmoid_activation
			);
}
