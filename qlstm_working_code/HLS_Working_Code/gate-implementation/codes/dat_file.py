import numpy as np

data_array = np.ones([20,1])
np.savetxt("weights.dat",data_array,fmt='%d')