#include "gate-mt-header.h"
#include <cstdio>

int main(){

	inp_t X[Inp_N];
	inp_t H[Out_N];
	inp_t C[Out_N];
	inp_t G_1[Out_N];
	inp_t G_2[Out_N];
	inp_t G_3[Out_N];
	inp_t C_U[Out_N];
	inp_t C_F[Out_N];
	inp_t H_F[Out_N];

	sigmoid_t sigmoid_table[Act_N];
	tanh_t tanh_table[Act_N];

	int i,j;
	//Input Values
	for(i=0;i<Inp_N;i++){
		X[i] = 0;
	}
	X[0] = 2;
	//Initial hidden and cell states
	for(i=0;i<Out_N;i++){
			H[i] = 0;
			C[i] = 0;
	}

	//Here I am passing the weight matrix, the recurrent matrix and the bias vector as inputs to the gate function.
	//Will now be able to call multiple instances of this function on for the four different gates that we require these for.
	inp_t weight[Out_N][Inp_N];
	inp_t recurrent[Out_N][Out_N];
	inp_t bias[Out_N];

	//Tried to define the above weights parameters as static variables but I was not able to declare them as static types when passed as an input parameter to the required function.

	//Initialization loop
	int tmp = 0;
	FILE *fp;

	//Now save the weights from the onnx file which is stored as a numpy array and test the design with those values here.
	//Either this. Or test the graph with the fully float LSTM layer in onnx.
	fp=fopen("/home/khandelw/Desktop/codes/weights.dat","r");
	//Load weight matrix values.
	loop_weight_1: for(i=0;i<Out_N;i++){
		loop_weight_2: for(j=0;j<Inp_N;j++){
			fscanf(fp,"%d",&tmp);
			weight[i][j] = 1;
			//weight[i][j] = tmp;
		}
	}
	fclose(fp);
	//Load recurrent matrix values.
	loop_recurrent_1: for(i=0;i<Out_N;i++){
		loop_recurrent_2: for(j=0;j<Out_N;j++){
				recurrent[i][j] = 1;
			}
		}
	//Load bias values.
	loop_bias_intermediate: for(i=0;i<Out_N;i++){
		bias[i] = 0;
		}

	//LUT based 'sigmoid' activation function computation.
	//We can only have 256 values for activations as we are in the INT8 precision and the values in the computation can only range from  [-128, 127]. Only 256 possible inputs hence only 256 values for the activation function.
	fp=fopen("/scratch/users/khandelw/sigmoid.dat","r");
	for(int a = 0; a < Act_N; a++){
		fscanf(fp,"%d",&tmp);
		sigmoid_table[a] = tmp;
		cout << "Sigmoid value = " << tmp  << "\n";
	}
	fclose(fp);
	tmp = 0;
	//LUT based 'tanh' activation function computation.
	fp=fopen("/scratch/users/khandelw/tanh.dat","r");
    for (int ii = 0; ii < Act_N; ii++) {
    	fscanf(fp,"%d",&tmp);
        tanh_table[ii] = tmp;
        cout << "Tanh value = " << tmp  << "\n";
    }
    fclose(fp);

	//Top function LSTM call.
    gate_function_top(
			X,H,
			G_1,
			weight,
			recurrent,
			bias,
			sigmoid_table, tanh_table
	);

	for(i=0;i<Out_N;i++){
		cout << " Gate Out = " << G_1[i] << "\n";
	}
	return 0;
}
//"Gate-1 out = " << G_1[i] << " Gate-2 out = " <<  G_2[i] << " Gate-3 out = " <<  G_3[i] << " Cell Update Out = " <<  C_U[i]  <<
