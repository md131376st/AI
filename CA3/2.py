import numpy as np
import glob
from PIL import Image
import pickle
import random
import math
import matplotlib.pyplot as plt

c = 1
inputSize = 787
outputSize = 10
hiddenSize = 20


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None, hidden_layer_bias=None
                 , output_layer_weights=None, output_layer_bias=None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        # print (len(self.hidden_layer.neurons))
        # en(self.hidden_layer.neuron)

        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        print("start feed_forward")
        # print(inputs)
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        # print( hidden_layer_outputs)
        return self.output_layer.feed_forward(hidden_layer_outputs,False)

    def train(self, training_inputs, training_outputs):
        print("start train")
        self.feed_forward(training_inputs)
        print("finish feed_forward")
        # self.gradient_descent(training_outputs)
        self.gradient_descent_descent(training_outputs)

    def gradient_descent(self, training_outputs):
        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        # print(len(self.output_layer.neurons))
        print("output nuran detailes")
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = \
                self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
        # print(pd_errors_wrt_output_neuron_total_net_input)
        # 2. Hidden neuron deltas
        print("Hidden neuron deltas")
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * \
                                                    self.output_layer.neurons[o].weights[h]
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * \
                                                             self.hidden_layer.neurons[
                                                                 h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        print("Update output neuron weights")
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[
                    o].calculate_pd_total_net_input_wrt_weight1(w_ho)

                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        print("Update hidden neuron weights")
        for h in range(len(self.hidden_layer.neurons)):
            num=0
            for w_ih in range(28):
                for w_ihi in range(28):
                    pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[
                        h].calculate_pd_total_net_input_wrt_weight(w_ih, w_ihi)
                    self.hidden_layer.neurons[h].weights[num] -= self.LEARNING_RATE * pd_error_wrt_weight
                    num+=1

    def gradient_descent_descent(self, training_outputs):
        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        # print(len(self.output_layer.neurons))
        print("output nuran detailes")
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = \
                self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
        # print(pd_errors_wrt_output_neuron_total_net_input)
        # 2. Hidden neuron deltas
        print("Hidden neuron deltas")
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] *\
                                                    self.output_layer.neurons[o].weights[h]
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output *\
                                                             self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        print("Update output neuron weights")
        for o in range(len(self.output_layer.neurons)):
            pd_error_wrt_weight=0
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight += pd_errors_wrt_output_neuron_total_net_input[o] *\
                                       self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight1(w_ho)
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight \
                                                              / len(self.output_layer.neurons[o].weights)

        # 4. Update hidden neuron weights
        print("Update hidden neuron weights")
        for h in range(len(self.hidden_layer.neurons)):
            pd_error_wrt_weight =0
            for w_ih in range(28):
                for w_ihi in range(28):
                    pd_error_wrt_weight += pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[
                        h].calculate_pd_total_net_input_wrt_weight(w_ih, w_ihi)
            for i in range(len(self.hidden_layer.neurons[h].weights)):
                    self.hidden_layer.neurons[h].weights[i] -= self.LEARNING_RATE * pd_error_wrt_weight/len(self.hidden_layer.neurons[h].weights)

    def calculate_total_error(self, training_sets):
        total_error = 0
        # for t in range(len(training_sets)):
            # print(training_sets[t])
        training_inputs, training_outputs = training_sets
        # print (training_inputs)
        self.feed_forward(training_inputs)
        for o in range(len(training_outputs)):
            total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs,numlayer=True):
        print("feed forward noron")
        outputs = []
        # print(len(self.neurons))
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs,numlayer))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs,numlayer=True):
        self.inputs = inputs
        self.output = self.squash_sigmoid(self.calculate_total_net_input(numlayer))
        # print (self.output)
        return self.output

    def calculate_total_net_input(self, layernum=True):
        total = 0
        num=0
        regularization=self.l2_norm()
        # regularization=0
        if layernum ==True:
            for i in range(28):
                for j in range(28):
                    total += (self.inputs[i][j] * self.weights[num])+regularization
                    num+=1
        else:
            for i in range(len(self.inputs)):
                total += self.inputs[i] * self.weights[i]+regularization
        return total + self.bias

    def l2_norm(self):
        num=0
        # print(self.weights)
        for i in range(len(self.weights)):
            num+=self.weights[i]*self.weights[i]
        # print("l2_nuem",num)
        return num * 0.005

    def drop_out(self):

        pass
    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash_sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def squash_linear(self, total_net_input):
        return total_net_input*c

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        # print("hi",self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input())
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        # print("******",(target_output - self.output)**2)
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_error_wrt_output(self, target_output):
        # print("target_output:", target_output, "output:",self.output)
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        # print("output:",self.output)
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index,index2):
        # print(self.inputs)
        return self.inputs[index][index2]

    def calculate_pd_total_net_input_wrt_weight1(self, index):
        return  self.inputs[index]


def creat_input():
    num = 0
    in_ = 'A'
    list_test=list()
    while num < 10:
        listname = glob.glob("./notMNIST_small/"+chr(ord(in_)+num)+"/*.png")
        list_ = list()
        for i in range(0, 500, 1):
            list_.append(np.asarray(Image.open(listname[i])))
        for i in range(5):
            list_test.append(np.asarray(Image.open(listname[random.randint(501, len(listname))])))

        pickle.dump(list_, open(chr(ord(in_)+num)+".pickle", "wb"))
        num += 1
    pickle.dump(list_test,open("test.pickle","wb"))
    pass
# creat_input()

def read_input():
    num = 0
    in_ = 'A'
    list__=list()
    while num < 10:
        list_ = pickle.load(open(chr(ord(in_)+num)+".pickle", "rb"))
        list__.append(list_)
        num+=1
    # print(le)
    test_list=pickle.load(open("test.pickle","rb"))
    return test_list,list__

creat_input()
test_list,list_ = read_input()
nn = NeuralNetwork(inputSize, hiddenSize, outputSize)
list_y=list()

for j in range(500):
    # tarinlist=list()
    for i in range(10):
        output = list()
        if i == 0:
            output = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # output=[1000000000]
        elif i == 1:
            output = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif i == 2:
            output = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif i == 3:
            output = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif i == 4:
            output = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif i == 5:
            output = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif i == 6:
            output = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif i == 7:
            output = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif i == 8:
            output = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif i == 9:
            output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        print("calling ",i+j*10)
        nn.train(list_[i][j],output)
        # nn.inspect()
        list_y.append(nn.calculate_total_error([list_[i][j],output]))
        # if i==0:break
    # if j==0:break
        # print(i+j, round(nn.calculate_total_error([list_[i][j],output])))
    # print(len(tarinlist))
plt.plot(list_y)
plt.show()
for j in range(len(test_list)):
    __list=nn.feed_forward(test_list[j])
    print(__list.index(max(__list)))
# input_=np.asarray(Image.open("./Q2FzbG9uVHdvVHdlbnR5Rm91ci1CbGFja0l0Lm90Zg==.png"))
# print(nn.feed_forward(input_))
# nn.inspect()

# for i in range(20):
#     nn.train(list_[i], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
