from base import *

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weight = np.random.normal(0, math.sqrt(2.0/(input_size + output_size)), (input_size, output_size))
        self.bias   = np.zeros(output_size)

        self.weight_velocity = np.zeros((input_size, output_size))
        self.bias_velocity   = np.zeros(output_size)

    def forward(self, input_value):
        self.saved_input = input_value
        return np.matmul(self.saved_input, self.weight) + self.bias

    def backward(self, output_gradient):
        input_gradient       = np.matmul(output_gradient, self.weight.transpose())
        self.weight_gradient = np.matmul(self.saved_input.transpose(), output_gradient)
        self.bias_gradient   = np.sum(output_gradient, 0)
        return input_gradient

    def train(self, learning_rate, momentum):
        self.weight_velocity = self.weight_velocity * momentum + self.weight_gradient * (1.0 - momentum)
        self.bias_velocity   = self.bias_velocity   * momentum + self.bias_gradient   * (1.0 - momentum)

        self.weight -= learning_rate * self.weight_velocity
        self.bias   -= learning_rate * self.bias_velocity

class ActivationLayer:
    def __init__(self):
        pass

    def forward(self, input_value):
        self.saved_input = input_value
        return np.tanh(input_value)

    def backward(self, output_gradient):
        return output_gradient / np.square(np.cosh(self.saved_input))

    def train(self, learning_rate, momentum):
        pass

class Layer:
    def __init__(self, input_size, output_size):
        self.dense      = DenseLayer(input_size, output_size)
        self.activation = ActivationLayer()

    def forward(self, input_value):
        return self.activation.forward(self.dense.forward(input_value))

    def backward(self, output_gradient):
        return self.dense.backward(self.activation.backward(output_gradient))

    def train(self, learning_rate, momentum):
        self.dense.train(learning_rate, momentum)
        self.activation.train(learning_rate, momentum)

class Model:
    def __init__(self):
        self.layer1 = Layer(6, 16)
        self.layer2 = Layer(16, 8)
        self.layer3 = Layer(8, 4)
        self.layer4 = Layer(4, 1)

        self.gradient_log = list()

    def forward(self, input_value):
        output_value = input_value
        output_value = self.layer1.forward(output_value)
        output_value = self.layer2.forward(output_value)
        output_value = self.layer3.forward(output_value)
        output_value = self.layer4.forward(output_value)
        return output_value

    def backward(self, output_gradient):
        input_gradient = output_gradient
        input_gradient = self.layer4.backward(input_gradient)
        input_gradient = self.layer3.backward(input_gradient)
        input_gradient = self.layer2.backward(input_gradient)
        input_gradient = self.layer1.backward(input_gradient)

        self.gradient_log.append(np.linalg.norm(input_gradient))
        return input_gradient

    def train(self, learning_rate, momentum):
        self.layer1.train(learning_rate, momentum)
        self.layer2.train(learning_rate, momentum)
        self.layer3.train(learning_rate, momentum)
        self.layer4.train(learning_rate, momentum)
