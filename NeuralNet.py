import numpy as np 

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def sigmoid_der(x) :
    return x*(1-x)

# input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output dataset
training_outputs = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
np.random.seed(1)

weights= 2 * np.random.random((3,1)) - 1

#print(np.dot(training_inputs,weights))


for x in range(10000):
    outputs=sigmoid(np.dot(training_inputs,weights))
    error=training_outputs- outputs
    adjusments=error * sigmoid_der(outputs)
    weights+=np.dot(training_inputs.T,adjusments)
print('weights after training: ')
print(weights)

print("Output After Training:")
print(outputs)    