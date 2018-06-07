# Udacity-CarND: Traffic Sign Classifier

### The goal of this project is to make a traffic sign classifier by deploying a convolutional neural network trained on a [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

- The steps taken to build the classifier are as follows:
- Download and load the pickle data (the data is already split in training, validation and test sets)
- Analyze the data and check distribution 
- Preprocess the data (convert to grayscale, apply contrast limited adaptive histogram equalization, augment the data by applying affine transformations, normalize the data before feeding it to the neural network
- Define and design the CNN architechture
- Train the network on training data and check traininig and validation accuracies (repeat the process until desired accuracies are achieved, >93% for the project and push for better accuracies by changing hyperparameters, changing CNN architechture)
- Test the model on the test data
- Test the model on 5 or more images downloaded from internet and view the softmax probabilites for top 3 image guesses for downloaded test images

### Dataset summary

The data is consisted of:
- 34,799 Training Images
- 4,410 Validation Images
- 12,630 Test Images
- The images are sized to 32x32 pixels and they are spread across 43 classes (43 traffic signs)

### Preprocessing the data

We are provided with images of size 32x32x3. I converted all the images to grayscale foremost to make working with the data a little faster as the no. of channel information is reduced and all the important feature information is still conserved.

The traning data is skewed and some of the classes have less than 10 times the images w.r.t to the class with highest number of images. Because we don't have a balanced data, the trained network over this data will be biased towards recognizing the image it has seen the most. One of the ways to deal with this issue is to remove the data from the classes having more data, but is that the wat to go? We are throwing away legit data points from our dataset. Why not add more data to the dataset for the underrepresented classes? So here, we iterate through all the classes and check if they have less images than the mean number of images in the training data set. If they are less than the mean number of images, we apply transformations on the images such as translate, rotate and shear (you can do a lot more such as changing brightness, changing pov, blur, fade etc) and produce the copies of data for that classes until we reach the mean.

After this point we just need to normalize the data before feeding it to the network which makes the convergence and calculations occuring in the CNN faster and accurate.

### The CNN architechture

I experimented two architechtures in this project. 
The first network is LeNet which was developed by Yann LeCun in 1998 for classifying the hand written digits. 

<image>

The second architechture is [Traffic Sign Recognition with MultiScale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun. The layers of the CNN are setup like as following:

1. 5x5 convolution (32x32x1 in, 28x28x6 out)
2. ReLU
3. 2x2 max pool (28x28x6 in, 14x14x6 out)
4. 5x5 convolution (14x14x6 in, 10x10x16 out)
5. ReLU
6. 2x2 max pool (10x10x16 in, 5x5x16 out)
7. 5x5 convolution (5x5x6 in, 1x1x400 out)
8. ReLu
9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
10. Concatenate flattened layers to a single size-800 layer
11. Dropout layer
12. Fully connected layer (800 in, 43 out)

![alt text](C:\Users\vzy75q\Desktop\ML\trafficpics/Drawing1.jpg)



