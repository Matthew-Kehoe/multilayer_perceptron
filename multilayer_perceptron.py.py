
import math
import numpy as np
import pandas as pd
from random import seed
from random import uniform

# NETWORK SETUP

def network_initalization(num_inputs, num_hidden, num_outputs):
    
    def initial_weight_generator(left_layer_units, right_layer_units):
        #generates weights between two perceptron layers, weights are randomly generated and kept in range [-1,1]
        weights = []
        for i in range(right_layer_units):
            single_neuron_weight = []

            for j in range(left_layer_units):
                single_neuron_weight.append(uniform(-1, 1))
            
            weights.append({"weights": single_neuron_weight})

        return weights

    network_layers= []
    network_layers.extend([initial_weight_generator(num_inputs, num_hidden), initial_weight_generator(num_hidden, num_outputs)])

    return network_layers

# FORWARD PROPAGTION SECTION

def neuron_activate(weights, inputs):

    activation = 0
    for weight in weights:
        weight
    return np.dot(weights, inputs)

def stable_sigmoid(x):
    #squashes input into 0-1 range
    """ sourced from https://www.delftstack.com/howto/python/sigmoid-function-python/"""
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


def activation_processor(unit_activation_value, activation_function):
    return activation_function(unit_activation_value)


def fr_prp(network, example):

    inputs = example


    for layer in network:
        inputs = layer_outputer(layer, inputs)

    return inputs


def layer_outputer(layer: list, inputs: list) -> list:
    """ Calculates the oupts for a given layer by dot producting the inputs by the weights and then squashing the result"""

    layer_output = []


    for neuron in layer:
        
        #calulating the raw activation for a given unit
        activation_value = neuron_activate(inputs, neuron['weights'])

        #applying squashing function
        output = activation_processor(activation_value, stable_sigmoid)
        #output = activation_processor(activation_value, tanh)
        neuron['out'] = output
        layer_output.append(output)
    
    return layer_output


# BACKPROPAGATION SECTION

def sigmoid_derivative(output):
    return output * (1.0 - output)

#tanh
def tanh(z):
	return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

def tanh_derivative(z):
	return 1 - tanh(z)**2

# Backpropagate error and store in neurons
def back_prop(network, expected):

    #iterating through layers backward
	for i in range(len(network)-1, -1, -1):

        #selecting current layer
		layer = network[i]
		error_list = []

        #calculation of error for all other layers
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error = error + (neuron['weights'][j] * neuron['del'])
				error_list.append(error)

		#handling the calculation of the error in the last layer
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				error_list.append(neuron['out'] - expected[j])
        

		for neuron, error in zip(layer, error_list):
			neuron['del'] = error * sigmoid_derivative(neuron['out'])
			#neuron['del'] = error * tanh_derivative(neuron['out'])


def weight_updater(network, inputs, learning_rate):
    """
    NEEDS TO BE CHANGED 
    """

    for i in range(len(network)):
        #slcing input array until last value (taget)

        #handling last layer weight update
        if i!=0:
            inputs = [neuron['out'] for neuron in network[i -1]]

        for neuron in network[i]:


            for j in range(len(inputs)):

                neuron['weights'][j] -= learning_rate * neuron['del'] * inputs[j]

                

def fit(network, train, l_rate, epochs, n_outputs):
    for epoch in range(epochs):
        
        #storing total error for backprop
        total_err = 0

        for row in train:

            inputs=row[0]

            target = [row[-1]]



            #outputs = forward_prop(network, row)
            outputs = fr_prp(network, inputs)
            #expected = [0 for i in range(n_outputs)]

            target = [row[-1]]

            total_err = total_err + pow((target[0]-outputs[0]), 2)
            back_prop(network, target)
            weight_updater(network, inputs, l_rate)

        if epoch % 50 == 0:
            #error_learning_value.append(total_err)
            print(f'Epoch={epoch}, error={total_err}')

            

def predict_single(network, input_array):
    network_output = fr_prp(network, input_array)

    return 1 * (network_output[0] > 0.5)



seed(1)
dataset = [[[0,0],0],[[0,1],1], [[1,0],1], [[1,1],0]]

number_inputs = 2
number_hidden = 4
number_outputs= 1


#code for outputting error based on learning rate
#error_learning_rate = []

# for i in range (15,35, 5):
#     error_learning_value = []
#     network = network_initalization(number_inputs, number_hidden, number_outputs)
#     fit(network, dataset, i/100, 5000 , number_outputs)
#     #error_learning_rate.append(error_learning_value)
# # for layer in network:
# #     print(layer)


#pd.DataFrame(error_learning_rate).to_csv("output.csv",index=False)


network = network_initalization(number_inputs, number_hidden, number_outputs)
fit(network, dataset, 0.20, 5000 , number_outputs)

print("\nFinalized neuron values: ")
for layer in network:
    print("\nLayer: ", layer)

print("\nFinalized predictions: ")
print("[0,0] predict example: ", predict_single(network, [0,0]))
print("[0,1] predict example: ", predict_single(network, [0,1]))
print("[1,0] predict example: ", predict_single(network, [1,0]))
print("[1,1] predict example: ", predict_single(network, [1,1]))
