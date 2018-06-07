# Udacity-CarND_Traffic_Sign_Classifier-Project2

## The goal of this project is to make a traffic sign classifier by deploying a convolutional neural network trained on a [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

- The steps taken to build the classifier are as follows:
- Download and load the pickle data (the data is already split in training, validation and test sets)
- Analyze the data and check distribution 
- Preprocess the data (convert to grayscale, apply contrast limited adaptive histogram equalization, augment the data by applying affine transformations, normalize the data before feeding it to the neural network
- Define and design the CNN architechture
- Train the network on training data and check traininig and validation accuracies (repeat the process until desired accuracies are achieved, >93% for the project and push for better accuracies by changing hyperparameters, changing CNN architechture)
- Test the model on the test data
- Test the model on 5 or more images downloaded from internet and view the softmax probabilites for top 3 image guesses for downloaded test images

## Dataset summary 

The data is consisted of 
- 34,799 Training Images
- 4,410 Validation Images
- 12,630 Test Images

The images are sized to 32x32 pixels and they are spread across 43 classes (43 traffic signs)
