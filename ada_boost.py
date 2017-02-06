from collections import Counter
from stump_details import Stump
from stump_details import Best_Stump
from random import randint
import math
import sys

''''
***************************AdaBoost**********************
The adaboost technique constructs the stump which contains 4 single node tree, one each for 1 vs all classifier for 0, 90, 180, 270 in each iteration.
Thus, separate weights are associated for each 1 vs all classifier.
After each iteration, the weight for each sample of training data, for each classifier is updated based on whether it was correctly classified or not
The weight is updated as:
weight = weight * exp(-alpha) if the sample is correctly classified.
weight = weight * exp(alpha) if the sample is incorrectly classified
here alpha is computed as:
alpha = 0.5 * log(1-error/ error)
Once the weights for all the samples are updated for a particular sample, normalization is done so that sum of all weights is 1.
Once the weights are updated, another stump is constructed using the updated weights.

***The number of attributes to be considered for training is hard coded to 4000. It can be changed in generate_pairs(lower_bound, upper_bound) method



Testing:
Once the training is completed, we have n best stumps, each consisting of 4 classifier stumps.
The testing is done on each classifier stump to find out if it is correctly classified then add the alpha of that classifier to the correct index of the predicted list.
This is done for all classifiers of all stumps.
After testing on all of them is done, the predicted list will contain the sum of alpha values of all the classifiers corresponding to their index.
Thus the final predicted value is the index in the predicted  list which has maximum value. The index will be one of 0 to 3 which corresponding to 0, 90, 180, 270.
This predicted value is compared with actual value to find whether it is correctly classified.
If correctly classified then the correct counter is incremented
'''



''''
#measure the performance of the stump with chosen condition(first_index < second_index) and weights given as input
for each training image check if the stump correctly classifies it with the given condition.
this check is performed for 0, 90, 180 and 270 rotation. Thus each stump has correct classified list and score for all the rotations.
score is computed by adding the weight of the image if it is correctly classified.
Weight is added to score corresponding to the rotation same as its actual rotation if correctly classified.
If the condition is false but the image's rotation doesn't match the rotation under consideration then it is considered as correctly classified


'''
def compute_stump(train_images, first_index, second_index, weights_0, weights_90, weights_180, weights_270):
    #print "Computing stumps!!!"
    stump = Stump()
    for i in range(len(train_images)):
        if int(train_images[i][first_index]) < int(train_images[i][second_index]):
            #print "Rotation is ", train_images[i][1]
            #raw_input()
            if int(train_images[i][1]) == 0:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_0.add(i)
                stump.score_0 += weights_0[i] * 1
            elif int(train_images[i][1]) == 90:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_90.add(i)
                stump.score_90 += weights_90[i] * 1
            elif int(train_images[i][1]) == 180:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_180.add(i)
                stump.score_180 += weights_180[i] * 1
            elif int(train_images[i][1]) == 270:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_270.add(i)
                stump.score_270 += weights_270[i] * 1
        else:
            if int(train_images[i][1]) == 0:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_90.add(i)
                stump.correct_classified_180.add(i)
                stump.correct_classified_270.add(i)
                stump.score_90 += weights_90[i] * 1
                stump.score_180 += weights_180[i] * 1
                stump.score_270 += weights_270[i] * 1
            elif int(train_images[i][1]) == 90:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_0.add(i)
                stump.correct_classified_180.add(i)
                stump.correct_classified_270.add(i)
                stump.score_0 += weights_0[i] * 1
                stump.score_180 += weights_180[i] * 1
                stump.score_270 += weights_270[i] * 1
            elif int(train_images[i][1]) == 180:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_90.add(i)
                stump.correct_classified_0.add(i)
                stump.correct_classified_270.add(i)
                stump.score_90 += weights_90[i] * 1
                stump.score_0 += weights_0[i] * 1
                stump.score_270 += weights_270[i] * 1
            elif int(train_images[i][1]) == 270:
                #print "Inside if"
                #raw_input()
                stump.correct_classified_90.add(i)
                stump.correct_classified_180.add(i)
                stump.correct_classified_0.add(i)
                stump.score_90 += weights_90[i] * 1
                stump.score_180 += weights_180[i] * 1
                stump.score_0 += weights_0[i] * 1
    stump.first_index = first_index
    stump.second_index = second_index
    return stump

def start_adaboost(train_images, test_images, weights):
    print "Adaboost computation started"
    #weights are initialized to 1/N if running for 1st stump
    if weights == None:
        #print "Weights were none!!!"
        weights = []
        for i in range(4):
            curr_weight = [float(1)/ float(len(train_images))] * len(train_images)
            weights.append(curr_weight)
        #weights = [[float(1)/ float(len(train_images))] * len(train_images)] *4
    #print "no of weights list", len(weights)
    #print "length of each weight list", len(weights[0])
    best_stumps = [None] *4

    # generate 4000 random pairs of features to consider as the attributes for stump condition
    pairs = generate_pairs(2, len(train_images[0])-1)
    print "Selected attributes!!!"
    #for i in range(2, len(train_images[0])):
    #    for j in range(2, len(train_images[0])):
    print "Training..... "
    for pair in pairs:
        i = pair[0]
        j = pair[1]
        # determine the correct classifications for different rotations using condition in current pair and current weights
        current_stump = compute_stump(train_images, i, j, weights[0], weights[1], weights[2], weights[3])
        #print "current stump score", current_stump.score_0, current_stump.score_90, current_stump.score_180, current_stump.score_270
        #raw_input()
        if best_stumps[0] == None or best_stumps[0].score < current_stump.score_0:
            best_stumps[0] = populate_best_stump(current_stump, 0)
        if best_stumps[1] == None or best_stumps[1].score < current_stump.score_90:
            best_stumps[1] = populate_best_stump(current_stump, 90)
        if best_stumps[2] == None or best_stumps[2].score < current_stump.score_180:
            best_stumps[2] = populate_best_stump(current_stump, 180)
        if best_stumps[3] == None or best_stumps[3].score < current_stump.score_270:
            best_stumps[3] = populate_best_stump(current_stump, 270)
    degree = 0
    weight_index = 0

    #best stumps contains the decision stump for each rotation that has best score.
    for stump in best_stumps:
        compute_alpha(stump)
        print "stump", weight_index, " alpha= ",stump.alpha
        ''''
        if degree ==0:
            correct_classified_list = stump.correct_classified_0
        elif degree == 90:
            correct_classified_list = stump.correct_classified_90
        elif degree == 180:
            correct_classified_list = stump.correct_classified_180
        elif degree == 270:
            correct_classified_list = stump.correct_classified_270
        '''
        compute_weights(stump.correct_classified_list, weights[weight_index], stump.alpha)
        degree += 90
        weight_index +=1
    print "training completed!!!"
    #print "weights len", len(weights)
    #print "weights", weights[0][0], weights[0][1], weights[1][0], weights[1][1], weights[2][0], weights[2][1]
    return best_stumps, weights

def generate_pairs(lower_bound, upper_bound):
    pairs = set()
    while len(pairs) < 4000:
        a = randint(lower_bound, upper_bound)
        b = randint(lower_bound, upper_bound)
        if a <> b:
            pair = (a, b)
            if pair not in pairs:
                pairs.add(pair)
    return pairs

#compute the alpha as (1-error)/ error where error = 1-score
def compute_alpha(stump):
    print "Compute alpha!!!"
    ''''
    if degree == 0:
        #error = 1- float(len(stump.correct_classified_0)) / float(total_records)
        error  = 1 - stump.score_0
    elif degree == 90:
        #error = 1 - float(len(stump.correct_classified_0)) / float(total_records)
        error = 1 - stump.score_90
    elif degree == 180:
        #error = 1 - float(len(stump.correct_classified_0)) / float(total_records)
        error = 1 - stump.score_180
    elif degree == 270:
        #error = 1 - float(len(stump.correct_classified_0)) / float(total_records)
        error = 1 - stump.score_270
    '''
    error = 1 - stump.score
    error_rate = (1 - error) / (error)
    alpha = 0.5 * math.log(error_rate)
    stump.alpha = alpha

# update weights with weight * exp(-alpha) if correctly classified
# update weights with weight * exp(alpha) if incorrectly classified
def compute_weights(correct_classified_list, weights, alpha):
    print "Compute weights!!!"
    total_weight = 0.0
    for i in range(len(weights)):
        if i in correct_classified_list:
            #print "Initial weight", weights[i]
            weights[i] = weights[i] * math.exp(-alpha)
            #print "Updated weight", weights[i]
            total_weight += weights[i]
        else:
            #print "Initial weight", weights[i]
            weights[i] = weights[i] * math.exp(alpha)
            #print "Updated weight", weights[i]
            #raw_input()
            total_weight += weights[i]
    #normalize weight
    for i in range(len(weights)):
        weights[i] /= total_weight


#populate the Best_Stump using the details in stump
def populate_best_stump(stump, degree):
    #print "Creating best stump object!!!"
    best_stump = Best_Stump()
    if degree ==0:
        best_stump.correct_classified_list = stump.correct_classified_0
        best_stump.score = stump.score_0
    elif degree == 90:
        best_stump.correct_classified_list = stump.correct_classified_90
        best_stump.score = stump.score_90
    elif degree == 180:
        best_stump.correct_classified_list = stump.correct_classified_180
        best_stump.score = stump.score_180
    if degree == 270:
        best_stump.correct_classified_list = stump.correct_classified_270
        best_stump.score = stump.score_270
    best_stump.first_index = stump.first_index
    best_stump.second_index = stump.second_index
    return best_stump

# perform test on one stump
def test_for_stump(stump, test_data):
    predicted = Counter()
    degree = 0
    for classifier in stump:
        if int(test_data[classifier.first_index]) < int(test_data[classifier.second_index]):
            predicted[degree] += classifier.alpha
        else:
            predicted[degree] +=0
        degree += 90
    return predicted

#get the index with max score of predict value
def get_prediction(predict):
    max = - (sys.maxint -1)
    max_index = -1
    for i in range(len(predict)):
        if predict[i] > max:
            max_index = i
            max = predict[i]
    return max_index

#print confusion matrix
def print_confusion_matrix(confusion_matrix):
    print "\n\t\t\t\tPredicted"
    print "\t\t\t0\t\t90\t\t180\t\t270"
    print "Actual: 0\t\t",confusion_matrix[0][0],"\t\t",confusion_matrix[0][1],"\t\t",confusion_matrix[0][2],"\t\t",confusion_matrix[0][3]
    print "Actual: 90\t\t", confusion_matrix[1][0], "\t\t", confusion_matrix[1][1], "\t\t", confusion_matrix[1][2], "\t\t", confusion_matrix[1][3]
    print "Actual: 180\t\t", confusion_matrix[2][0], "\t\t", confusion_matrix[2][1], "\t\t", confusion_matrix[2][2], "\t\t", confusion_matrix[2][3]
    print "Actual: 270\t\t", confusion_matrix[3][0], "\t\t", confusion_matrix[3][1], "\t\t", confusion_matrix[3][2], "\t\t", confusion_matrix[3][3]

# perform testing on stumps created.
def test_ada_boost(stumps, test_data):
    print "Testing Ada Boost!!!"
    correct = 0
    confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    allowed_labels = [0, 90, 180, 270]
    f = open('adaboost_output.txt', 'w')
    for test_sample in test_data:
        #predictions = Counter({0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0})
        predict = [0.0] *4
        index = 0
        for stump in stumps:
            #print "stump", index
            index +=1
            #predictions + test_for_stump(stump, test_sample)
            test_prediction = test_for_stump(stump, test_sample)
            predict[0] += test_prediction[0]
            predict[1] += test_prediction[90]
            predict[2] += test_prediction[180]
            predict[3] += test_prediction[270]
        predicted_index = get_prediction(predict)
        predicted = predicted_index * 90
        actual_index = int(test_sample[1])/ 90
        if predicted == int(test_sample[1]):
            correct +=1
        f.write(test_sample[0] + " " + str(predicted) + "\n")
        confusion_matrix[actual_index][predicted_index] += 1
    f.close()
    print_confusion_matrix(confusion_matrix)
    print "Accuracy = ", float(correct)/ float(len(test_data))
