/*
 * Copyright 2022 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _LOOP_PERFECT_H_
#define _LOOP_PERFECT_H_

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ap_int.h"
#include "ap_fixed.h"
//-------------------------------------------------------------------------------------

//Instead of define use 'constexpr' which is an assignment now. #define considered really bad for the compiler.
constexpr unsigned Inp_N = 10; //This factor defines the input length at each time step.
constexpr unsigned Out_N = 20; //This factor defines the number of LSTM cells in the LSTM layer.
constexpr unsigned Act_N = 256;

// LSTM variables
using inp_t = ap_int<8>;
using param_t = ap_int<8>;
using out_t = ap_int<8>;
using tanh_t = ap_int<8>;
using sigmoid_t = ap_int<8>;
using mul_t = ap_int<16>; //Used for the 5th and 6th equations but should technically be used for all equations. Will have to correct this.

//using act_t = ap_fixed<24,8>;
//18 is the length of the variable, 8 is the number of bits used to represent the integer value, AP_RND defines the quantization mode.
//Dictates behaviour when greater precision is generated than can be defined by the smallest fractional bit.

// Top function of the project is set in the project settings.
//void loop_perfect(din_t A[N], dout_t B[N]);
void pipelined_lstm_top(
		inp_t X[Inp_N],inp_t H[Out_N], inp_t C[Out_N],
		inp_t c_f[Out_N], inp_t h_f[Out_N],
		param_t weight_f[Out_N][Inp_N], param_t weight_i[Out_N][Inp_N], param_t weight_o[Out_N][Inp_N], param_t weight_c[Out_N][Inp_N],
		param_t recurrent_f[Out_N][Out_N], param_t recurrent_i[Out_N][Out_N], param_t recurrent_o[Out_N][Out_N], param_t recurrent_c[Out_N][Out_N],
		param_t bias_f[Out_N], param_t bias_i[Out_N], param_t bias_o[Out_N], param_t bias_c[Out_N],
		sigmoid_t sigmoid_activation[Act_N], tanh_t tanh_activation[Act_N]
);

void gate_function(
		inp_t X[Inp_N], inp_t H[Out_N],
		inp_t g[Out_N],
		param_t weight[Out_N][Inp_N], param_t recurrent[Out_N][Out_N], param_t bias[Out_N],
		sigmoid_t sigmoid_gate[Act_N]
);

void cell_update_function(inp_t X[Inp_N],inp_t H[Out_N],
		inp_t g[Out_N],
		param_t weight[Out_N][Inp_N],param_t recurrent[Out_N][Out_N], param_t bias[Out_N],
		tanh_t tanh_gate[Act_N]
);

void cell_function(
		inp_t g_f, inp_t C, inp_t g_i, inp_t c_u, inp_t c_f
);
void hidden_function(
		tanh_t tanh_table[Act_N],inp_t c_f[Out_N], inp_t g_o[Out_N], inp_t h_f[Out_N]
);

void sigmoid_activation_function(
		out_t G_out[Out_N], sigmoid_t sigmoid_gate[Act_N]
);
void tanh_activation_function(
		out_t G_out[Out_N], tanh_t tanh_gate[Act_N], out_t G_out_act[Out_N]
);
void max_value(
		mul_t val, out_t *final_val
);

#endif

