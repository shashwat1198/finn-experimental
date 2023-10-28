//This is a single LSTM layer that is implemented in this HLS design.
//The design is a 'Static' variation where each input in a sequence is processed after the completion of the current input processing (II = Latency of a single input processing).
//The design can be made 'Non-Static' (utilizing parallelism across sequences) by instantiing multiple such blocks together. (II = 1 can be achieved.)

#include "pipeline-lstm-header.h"
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

void hidden_function(tanh_t tanh_table[Act_N],inp_t c_f[Out_N], inp_t g_o[Out_N], out_t h_f[Out_N]){
	out_t c_f_activation[Out_N];
	ap_int<16> temp = 0;
	ap_int<8> final_val;
	tanh_activation_function(c_f,tanh_table,c_f_activation);
	for(int j=0;j<1;j++){
		cout << "Tanh after cell state = " << c_f_activation[j] << "\n";
	}
	int i;
	loop_hidden_function:for(i=0;i<Out_N;i++){
		temp = c_f_activation[i] * g_o[i];
		//cout << "Final Hidden state = " << temp << "\n";
		max_value(temp,&final_val); //Restricting max value of the elementwise multiplier to 127. (Hardamad product)
		h_f[i] = final_val;
	}
	for(int k=0;k<1;k++){
			cout << "Final Hidden state = " << h_f[k] << "\n";
		}
}

void cell_function( inp_t g_f[Out_N], inp_t prev_C[Out_N], inp_t g_i[Out_N], inp_t c_u[Out_N], inp_t c_f[Out_N]){
	int i,j,k;
	ap_int<8> el_mul_1[Out_N];
	ap_int<8> el_mul_2[Out_N];
	ap_int<16> temp_1 = 0;
	ap_int<16> temp_2 = 0;
	ap_int<16> temp_3 = 0;
	ap_int<8> final_val_1;
	ap_int<8> final_val_2;
	ap_int<8> final_val_3;
	loop_cell_1:for(i=0;i<Out_N;i++){
		temp_1 = g_f[i] * prev_C[i];
		max_value(temp_1,&final_val_1);
		el_mul_1[i] = final_val_1;
	}
//sum datatype ap_int<16> : Need to quantize it back to 8-bits.
//	cout << "Sum 1 = " << el_mul_1[0] << "\n";
	loop_cell_2:for(j=0;j<Out_N;j++){
		temp_2 = g_i[j] * c_u[j];
		max_value(temp_2,&final_val_2);
		el_mul_2[j] = final_val_2;
	}
//	cout << "Sum 2 = " << el_mul_2[0] << "\n";
	loop_add_3:for(k=0;k<Out_N;k++){
		temp_3 = el_mul_1[k] + el_mul_2[k];
		max_value(temp_3,&final_val_3);
		c_f[k] = final_val_3;
	}
	cout << "Cell state = " << c_f[0] << "\n";
}

void cell_update_function(inp_t X[Inp_N],inp_t H[Out_N], inp_t cell_update[Out_N], param_t weight[Out_N][Inp_N],param_t recurrent[Out_N][Out_N], param_t bias[Out_N], tanh_t tanh_gate[Act_N]){
	int i,j;
	inp_t mv_1[Out_N];
	inp_t mv_2[Out_N];
	out_t cell_out[Out_N];

	loop_bias_intermediate: for(i=0;i<Out_N;i++){
		mv_1[i] = 0;
		mv_2[i] = 0;
	}

#pragma HLS ARRAY_PARTITION variable = weight dim = 1 complete
	loop_mv1_1_cu:for(i=0;i<Out_N;++i){
//#pragma HLS PIPELINE II=1
		loop_mv1_2_cu:for(j=0;j<Inp_N;++j){
			mv_1[i] = mv_1[i] + weight[i][j]*X[j];
	}
	}

#pragma HLS ARRAY_PARTITION variable = recurrent dim = 1 complete
//Second Matrix-Vector Multiply
	loop_mv2_1_cu:for(i=0;i<Out_N;i++){
//#pragma HLS PIPELINE II=1
		loop_mv2_2:for(j=0;j<Out_N;j++)
			mv_2[i] = mv_2[i] + recurrent[i][j]*H[j];
	}

//Addition operation
	loop_add_cu:for(i=0;i<Out_N;i++){
		cell_out[i] = mv_1[i]+mv_2[i]+bias[i];
	}
	tanh_activation_function(cell_out,tanh_gate,cell_update);
//	for(i=0;i<1;i++){
//		cout << "Tanh_loop = " << cell_update[i] << "\n";
//	}
//PRAGMA's specified : ARRAY PARTITION
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

void pipelined_lstm_top(
		inp_t X[Inp_N],inp_t H[Out_N], inp_t C[Out_N],
		inp_t c_f[Out_N], inp_t h_f[Out_N],
		param_t weight_f[Out_N][Inp_N], param_t weight_i[Out_N][Inp_N], param_t weight_o[Out_N][Inp_N], param_t weight_c[Out_N][Inp_N],
		param_t recurrent_f[Out_N][Out_N], param_t recurrent_i[Out_N][Out_N], param_t recurrent_o[Out_N][Out_N], param_t recurrent_c[Out_N][Out_N],
		param_t bias_f[Out_N], param_t bias_i[Out_N], param_t bias_o[Out_N], param_t bias_c[Out_N],
		sigmoid_t sigmoid_activation[Act_N], tanh_t tanh_activation[Act_N]
){
//Intermediate variables
	out_t g_f_1[Out_N], g_i_1[Out_N] ,g_o_1[Out_N], c_u_1[Out_N];

	gate_function(
			X, H,
			g_f_1, weight_f, recurrent_f , bias_f,
			sigmoid_activation
			);
	gate_function(
			X, H,
			g_i_1, weight_i, recurrent_i , bias_i,
			sigmoid_activation
			);
	gate_function(
			X, H,
			g_o_1, weight_o, recurrent_o ,bias_o,
			sigmoid_activation
			);
//This functions corresponds to the intermediate cell state update calculation.
	cell_update_function(
			X, H,
			c_u_1, weight_c, recurrent_c , bias_c,
			tanh_activation
			);
//This function calculates the final cell state for the given input.
	cell_function(
			g_f_1, C, g_i_1, c_u_1, c_f
			);
//This function calculates the hidden state for the given input.
	hidden_function(
			tanh_activation, c_f, g_o_1, h_f
	);
}
