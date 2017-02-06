'''
References :
http://staff.itee.uq.edu.au/janetw/cmc/chapters/BackProp/index2.html
http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

Used Numpy for all the matrix multiplications and other operations
We normalized the pixel data by dividing the pixel values by 255.
Used Stochastic method for training and shuffled the order of training for every iteration.

Command to run :  python orient.py train-data.txt test-data.txt nnet 20
Runs the code for 10 iterations and uses the learning rate of 0.2.

Prints the confusion matrix :
Hidden node count = 192
Confusion matrix::
	0   90   180  270
0	203	11	24	1
90	36	160	17	11
180	59	10	162	5
270	44	28	13	159
Accuracy : 0.725344644751 ( 684  out of  943 )

Highest accuracy achieved was 74% for 10 iterations, learning rate of 0.1 and hidden count = 20.
'''

import sys
import numpy as np
import random


def train_nn(nn, input_data):
    print "Training the model and calibrating weights !!"
    for i in range(0, 10):
        print "Shuffling Samples..."
        random.shuffle(input_data)
        print "Iteration :", i + 1
        for image in input_data:
            input_x = np.asmatrix(np.array(image.pixel_data / 255))
            current_output = nn.feed_forward(input_x)
            expected_output = [0] * 4
            expected_output[image.label / 90] = 1
            expected_output = np.asmatrix(expected_output)
            delta2 = nn.calculate_delta2(current_output, expected_output)
            delta1 = nn.calculate_delta1(delta2, nn.ah, nn.w2)
            # Update the weights
            nn.w2 = nn.w2 + np.multiply(np.dot(nn.ah.T, delta2), nn.learning_rate)
            nn.w1 = nn.w1 + np.multiply(np.dot(input_x.T, delta1), nn.learning_rate)


def read_file(file_name):
    images = []
    for line in open(file_name):
        temp = line.rstrip('\n').split(" ")
        images.append(ImageData(temp[0], int(temp[1]), np.array(map(int, temp[2:]), dtype=float)))
    return images


def test_nn(nn, test_data):
    print "Testing Accuracy!!"
    allowed_labels = [0, 90, 180, 270]
    correct_predictions = 0
    confusion_matrix = [[0 for x in range(4)] for y in range(4)]
    f = open('nnet_output.txt', 'w')
    for image in test_data:
        actual_label = image.label
        prediction = nn.feed_forward(np.array(image.pixel_data / 255))
        predicted_label = allowed_labels[np.argmax(prediction)]
        confusion_matrix[actual_label / 90][predicted_label / 90] += 1
        f.write(image.name + " " + str(predicted_label) + "\n")
        if predicted_label == actual_label:
            correct_predictions += 1
    f.close()
    print_confusion_matrix(confusion_matrix)
    print "Accuracy :", float(correct_predictions) / len(test_data), "(", correct_predictions, " out of ", len(
        test_data), ")"


def print_confusion_matrix(confusion_matrix):
    print "\nPrinting confusion matrix::"
    print '\t0   90   180  270'
    for row in range(len(confusion_matrix)):
        print str(row * 90) + "\t", ('\t'.join(map(str, confusion_matrix[row])))


class NeuralNetwork(object):
    input_layer_nodes = 0
    output_layer_nodes = 0
    hidden_layer_nodes = 0
    w1 = None
    w2 = None
    ao = None
    learning_rate = 0.1
    hidden_input = None
    ah = None
    hidden_output = None

    def __init__(self, input_layer_size, output_layer_size, hidden_layer_size):
        self.input_layer_nodes = input_layer_size
        self.output_layer_nodes = output_layer_size
        self.hidden_layer_nodes = hidden_layer_size
        print "Randomly assigning weights!!"
        self.w1 = np.random.randn(self.input_layer_nodes, self.hidden_layer_nodes)
        self.w2 = np.random.randn(self.hidden_layer_nodes, self.output_layer_nodes)
        print "Done."

    def feed_forward(self, input_x):
        self.hidden_input = np.dot(input_x, self.w1)
        self.ah = self.sigmoid(self.hidden_input)
        self.hidden_output = np.dot(self.ah, self.w2)
        self.ao = self.sigmoid(self.hidden_output)
        return self.ao

    # Apply Activation function - Using Sigmoid Activation function
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(np.negative(x)))

    @staticmethod
    def derivative_sigmoid(x):
        return np.multiply(x, (1.0 - x))

    @staticmethod
    def calculate_delta2(current_output, expected):
        # Reversing the Sign
        delta1 = np.multiply(np.subtract(expected, current_output), NeuralNetwork.derivative_sigmoid(current_output))
        return np.asmatrix(delta1)

    @staticmethod
    def calculate_delta1(delta1, ah, w2):
        return np.multiply(np.dot(delta1, w2.T), NeuralNetwork.derivative_sigmoid(ah))


def main(train_file, test_file, count):
    print "Reading Training Data!!"
    train_data = read_file(train_file)
    test_data = read_file(test_file)
    print "Initializing Neural Network!!"
    nn = NeuralNetwork(192, 4, int(count))
    train_nn(nn, train_data)
    test_nn(nn, test_data)


class ImageData(object):
    name = None
    label = 0
    pixel_data = []

    def __init__(self, name, label, pixel_data):
        self.name = name
        self.label = label
        self.pixel_data = pixel_data


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
