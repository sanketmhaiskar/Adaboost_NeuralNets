'''
Command: orient.py [train-path] [test-path] [nearest] 10
In this program we take input path for training data and test data along with the algorithm 'nearest' to run the code.

Algorithm:
1. Load the training data in the form of a matrix where every row in a matrix is a training example
2. Given a test example, calculate the Euclidean Distance of the test vector from all the train vectors
3. Find the train vector that has minimum Euclidean distance from the train vector
4. Assign the class of this nearest train vetor to the test image. (We use 1 nearest vector as in this KNN algorithm we use k = 1)

Results:
Confusion matrix::
	     0   90   180  270
--- |-------------------
0   |	155	24	36	24
90  |	19	154	19	32
180 |	38	28	154	16
270 |	20	33	20	171

Accuracy :: 67.23%
'''

import scipy.spatial.distance as scipy_dist
import numpy as np
import sys


train_data = {}
test_data = {}
nearest_neighbour = {}
test_files = []
confusion_matrix = [[0 for x in range(4)] for y in range(4)]


def getDistance(v1, v2):
    distance = scipy_dist.euclidean(v1,v2)
    return distance


def generate_image_matrix(img_pixel_matrix):
    image_dist_matrix = [[0.0 for x in range(len(img_pixel_matrix))] for x in range(len(img_pixel_matrix))]
    for i in range(len(img_pixel_matrix)-1):
        for j in range(i+1, len(img_pixel_matrix)):
            image_1 = []
            image_2 = []
            if i != j and img_pixel_matrix[i][j]== 0.0:
                image_1 = img_pixel_matrix[i]
                image_2 = img_pixel_matrix[j]
                getDistance(image_1,image_2)
                image_dist_matrix[i][j] = image_dist_matrix[j][i] = getDistance(image_1, image_2)
    print 'Distance matrix is ready!'


def load_data(path):
    img_pixel_matrix = []

    with open(path, 'r') as fp:
        i = 0

        for line in fp:
            vector = []
            curr_line = line.split(' ')
            train_data[i] = curr_line[1]
            for number in curr_line[2:]:
                if number.__contains__('\n'):
                    number = number.strip('\n')
                if number.__contains__(','):
                    number = number.strip(',')
                vector.append(int(number))
            img_pixel_matrix.append(vector)

            i += 1
    print 'Train data loaded.'
    fp.close()
    return img_pixel_matrix

def load_test_data(path, img_pixel_matrix):
    distances = []
    i = 0
    with open(path, 'r') as fp:
        for line in fp:
            min_dist = sys.maxint
            curr_line = line.split(' ')
            test_vector = []
            test_files.append(curr_line[0])
            for number in curr_line[2:]:
                if number.__contains__('\n'):
                    number = number.strip('\n')
                test_vector.append(int(number))
            test_data[i] = curr_line[1]
            j = 0
            for img_vector in img_pixel_matrix:
                dist = getDistance(test_vector, img_vector)
                if dist < min_dist and dist != 0:
                    min_dist = dist
                    nearest_neighbour[i] = j
                j += 1
            i += 1
    print 'Test data loaded.'
    fp.close()
    return nearest_neighbour, test_data

def print_confusion_matrix(confusion_matrix):
    print "\nPrinting confusion matrix::"
    print '\t0   90   180  270'
    for row in range(len(confusion_matrix)):
        print str(row * 90)+"\t", ('\t'.join(map(str,confusion_matrix[row])))


def find_accuracy():
    total_predictions = 0
    correct_predictions = 0
    incorrect_predictions = 0

    output = open("nearest_output.txt", 'w')
    for key in nearest_neighbour:
        predicted_class = train_data[nearest_neighbour[key]]
        actual_class = test_data[key]
        confusion_matrix[int(actual_class)/90][int(predicted_class)/90] += 1
        total_predictions += 1
        if predicted_class == actual_class:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
        output.write(test_files[key] + " "+ predicted_class +"\n")
    print_confusion_matrix(confusion_matrix)
    output.close()
    print 'Accuracy ::', (float(correct_predictions)/total_predictions)*100, ' %'




def find_nearest(train_path, test_path):
    img_pixel_matrix = load_data(train_path)
    print "Testing data..."
    nearest_neighbour, test_data = load_test_data(test_path, img_pixel_matrix)
    find_accuracy()
