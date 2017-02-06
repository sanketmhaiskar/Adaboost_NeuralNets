from ada_boost import start_adaboost
from ada_boost import test_ada_boost
import random
import sys
import nearest
import nnet

(train_path, test_path, algo) = sys.argv[1:4]
if algo != "best" and algo!="nearest":
    count = sys.argv[4]

#read the data from train and test file  into list
def read_data(train_file, test_file):
    print "Loading Train images!!!"
    with open(train_file, "r") as filehandler:
        train_images = [line.rstrip("\n").split() for line in filehandler]
        #image_details = filehandler.read().split()
        #images.append(image_details)
        #print "total images = ",len(train_images)
        #print "Dimensions for each image", len(train_images[0])

    print "Loading test images"
    with open(test_file, "r") as filehandler:
        test_images = [line.rstrip("\n").split() for line in filehandler]
        #image_details = filehandler.read().split()
        #images.append(image_details)
        #print "total images = ",len(test_images)
        #print "Dimensions for each image", len(test_images[0])
    return train_images, test_images

#contains code to divide training file into train file with 90% records and test file with 10% records.
def cross_validation(train_images):
    random.shuffle(train_images)
    train_size = int(len(train_images) *0.9)
    train_imgs = train_images[:train_size]
    test_imgs = train_images[train_size:]
    return train_imgs, test_imgs

if algo == "nearest":
    nearest.find_nearest(train_path, test_path)

if algo == "nnet":
    nnet.main(train_path, test_path,count)

if algo == "best":
    nnet.main(train_path, test_path, 20)
    
if algo == "adaboost":
    # additional_param, model_path, mode = sys.argv[4:7]
    train_images, test_images = read_data(train_path, test_path)
    # start_time = time()
    weights = None
    stumps = []
    # cross validation
    # train_img, test_img = cross_validation(train_images)
    # print "length of train data: ", len(train_images)
    # print "length of test data: ", len(test_images)
    for i in range(int(count)):
        best_stumps, weights = start_adaboost(train_images, test_images, weights)
        # print_stump(best_stumps)
        # raw_input()
        stumps.append(best_stumps)
    test_ada_boost(stumps, test_images)
