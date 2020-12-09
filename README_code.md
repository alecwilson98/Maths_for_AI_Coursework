**Instructions for running the linear regression models:**

Ensure that the packages numpy and pandas are installed on the python interpreter, and
the file 'Admission_predict_Ver1.csv' is downloaded. 

**Batch Gradient Descent model:**

The hyper-parameters for the model can be adjusted in the BGD() class within the arguments of the init() function. The
number of iterations, and the learning rate can be adjusted.

**Stochastic Gradient Descent model:**

The hyper-parameters for the model can be adjusted in the SGD() class within the arguments of the init() function.
The number of Epochs and the variables t0 and t1, which determine the starting point of the learning schedule, can be
adjusted. The equation which explains the learning schedule is found in the function learning_schedule().

**Mini-Batch Gradient descent model:**

The hyper-parameters for the model can be adjusted in the MBGD() class within  the arguments of the init() function. The
number of iterations, the learning rate, and the mini-batch size can be adjusted.



