import numpy as np
import struct


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051

        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows * cols)

    return images


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


class NeuralNetwork:
    class Layer:
        def __init__(self,sizeFrom,size,activeFunction=None):
            self.sizeFrom = sizeFrom
            self.size = size
            self.weights = np.random.randn(sizeFrom,size) * np.sqrt(2.0 / sizeFrom)
            self.bias = np.zeros(size)
            self.activeFunction = activeFunction

        def activation_function(self,z):
            if self.activeFunction == "ReLU":
                return np.maximum(z,0)
            if self.activeFunction == "Sigmoid":
                return sigmoid(z)
            if self.activeFunction == "tanh":
                return np.tanh(z)
            if self.activeFunction == "Softmax":
                return softmax(z)
            return z

        def derivative_activation_function(self,z):
            if self.activeFunction == "ReLU":
                return (z > 0).astype(float)
            if self.activeFunction == "Sigmoid":
                return sigmoid(z) * (1 - sigmoid(z))
            if self.activeFunction == "tanh":
                return 1 - np.tanh(z)**2
            if self.activeFunction == "Softmax":
                a = softmax(z)
                return np.diag(a) - np.outer(a, a)
            return z

    class MLP:
        def __init__(self,layers,costFunction,learningRate=0.01):
            self.layers = layers
            self.size = self.getsize()
            self.activations = None
            self.preActivations = None
            self.costFunction = costFunction
            self.learningRate = learningRate

        def getsize(self):
            arr = []
            is_first_layer = True
            for i in self.layers:
                if is_first_layer:
                    arr.append(i.sizeFrom)
                    is_first_layer = False
                arr.append(i.size)
            return arr

        def __len__(self):
            return len(self.layers)

        def _forward_pass(self,inputs,layer=None,currentLayer=None):
            if currentLayer is not None:
                layer=self.layers[currentLayer]
            if type(inputs) is int:
                inputs = [inputs]
            z = inputs @ layer.weights + layer.bias
            return z

        def forward_pass(self,inputs):
            self.activations = []
            self.preActivations = []
            self.preActivations.append(np.array(inputs))
            for i in self.layers:
                self.activations.append(np.array(inputs))
                z = self._forward_pass(inputs,layer=i)
                self.preActivations.append(np.array(z))
                inputs = i.activation_function(z)
            self.activations.append(np.array(inputs))
            return inputs

        #def dir_cost(self,pred_y,y):
            #if self.costFunction == "Cross-Entropy":
                #return -y / pred_y

        def _backward_pass(self,error,weights,activationDerivative):
            a = np.dot(weights, error) * activationDerivative
            return a


        def backward_pass(self,pred_y,y):
            errors = []
            for i in range(len(self.layers)-1,-1,-1):
                layer = self.layers[i]
                z = self.preActivations[i+1]
                if i == len(self.layers)-1:
                    if layer.activeFunction == "Softmax" and self.costFunction == "Cross-Entropy":
                        errors.append(pred_y - y)
                    else:
                        errors.append(np.dot(layer.derivative_activation_function(z), self.dir_cost(pred_y,y)))
                else:
                    lastLayer = self.layers[i+1]
                    activeDir = layer.derivative_activation_function(z)
                    errors.append(self._backward_pass(errors[-1],lastLayer.weights,activeDir))
            errors.reverse()
            return errors

        def calc_gradients(self,errors):
            delta_w = [None]*len(self.layers)
            delta_b = errors
            for i in range(len(self.layers)):
                delta_w[i] = self.activations[i].reshape(-1, 1) @ errors[i].reshape(-1, 1).T

            return delta_w, delta_b

        def _gradients(self, y, inputs):
            pred = self.forward_pass(inputs)
            err = self.backward_pass(pred,y)
            tmp = self.calc_gradients(err)
            delta_w, delta_b = tmp[0], tmp[1]
            return delta_w, delta_b

        def batch_gradients(self, ys, input_list):
            delta_wsum = None
            delta_bsum = None
            for i in range(len(ys)):
                tmp = self._gradients(ys[i],input_list[i])
                if i == 0:
                    delta_wsum = tmp[0]
                    delta_bsum = tmp[1]
                else:
                    delta_wsum += tmp[0]
                    delta_bsum += tmp[1]
            n = len(delta_wsum)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                layer.weights -= self.learningRate / n * delta_wsum[i]
                layer.bias -= self.learningRate / n * delta_bsum[i]
            return True

        def SDG_batch(self, ys, input_list, batch_size, batch_number):
            n = len(ys)
            for i in range(batch_number):
                for j in range(1,101):
                    if i == int(batch_number * .01 *j):
                        print(f"{j}% done")
                index = np.random.randint(0, n, size=batch_size)
                input_batch = input_list[index]
                y_batch = ys[index]
                self.batch_gradients(y_batch, input_batch)
            return True

        def accuracy(self,ys, input_list):
            correct = 0
            for i in range(len(ys)):
                guess = self.forward_pass(input_list[i])
                if np.argmax(guess) == np.argmax(ys[i]):
                    correct +=1

            return correct / len(ys)


# Initialize Neural net
l1 = NeuralNetwork.Layer(784,128, activeFunction="ReLU")
l3 = NeuralNetwork.Layer(128,128, activeFunction="ReLU")
l4 = NeuralNetwork.Layer(128,10, activeFunction="Softmax")
Mlp = NeuralNetwork.MLP([l1,l3,l4],"Cross-Entropy", learningRate = .001)


images = load_mnist_images("./t10k-images.idx3-ubyte")
image = images / 255
labels = load_mnist_labels("./t10k-labels.idx1-ubyte")
label = []
for i in range(len(labels)):
    temp = [0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]
    temp[labels[i]] = 1
    label.append(temp)

print(Mlp.accuracy(label,image))


images = load_mnist_images("./train-images.idx3-ubyte")
image = images / 255
labels = load_mnist_labels("./train-labels.idx1-ubyte")
label = []
for i in range(len(labels)):
    temp = [0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]
    temp[labels[i]] = 1
    label.append(temp)
Mlp.SDG_batch(np.array(label),np.array(image),4,120000)

images = load_mnist_images("./t10k-images.idx3-ubyte")
image = images / 255
labels = load_mnist_labels("./t10k-labels.idx1-ubyte")
label = []
for i in range(len(labels)):
    temp = [0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]
    temp[labels[i]] = 1
    label.append(temp)
print(Mlp.accuracy(label,image))
