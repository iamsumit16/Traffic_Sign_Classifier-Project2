# Udacity-CarND: Traffic Sign Classifier

### Overview
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the German Traffic Sign Dataset. After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

### Dependencies
This project requires Python 3.5 and the following Python libraries installed:

- Jupyter
- NumPy
- SciPy
- scikit-learn
- TensorFlow
- Matplotlib
- Pandas (Optional)


### The goal of this project is to make a traffic sign classifier by deploying a convolutional neural network trained on a [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The steps taken to build the classifier are as follows:
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

![Images from the German Traffic Sign Dataset](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/data_explore.png)

### Preprocessing the data

We are provided with images of size 32x32x3. I converted all the images to grayscale foremost to make working with the data a little faster as the no. of channel information is reduced and all the important feature information is still conserved.

![Converting images from RGB to Grayscale](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/data_explore.png)

The traning data is skewed and some of the classes have less than 10 times the images w.r.t to the class with highest number of images. 

![Data Distribution](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/data_distribution.png)

Because we don't have a balanced data, the trained network over this data will be biased towards recognizing the image it has seen the most. One of the ways to deal with this issue is to remove the data from the classes having more data, but is that the wat to go? We are throwing away legit data points from our dataset. Why not add more data to the dataset for the under-represented classes? So here, we iterate through all the classes and check if they have less images than the mean number of images in the training data set. If they are less than the mean number of images, we apply transformations on the images such as translate, rotate and shear (you can do a lot more such as changing brightness, changing pov, blur, fade etc) and produce the copies of data for that classes until we reach the mean.
Augmenting data resulted in over 46,000 training images.

![Applying transformation to existing data to create copies](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/transform.png)

![Data distribution after data augmentation](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/after_aug.png)

After this point we just need to normalize the data before feeding it to the network which makes the convergence and calculations occuring in the CNN faster and accurate.

### The CNN architecture

I experimented two architectures in this project. 
The first network is LeNet which was developed by Yann LeCun in 1998 for classifying the hand written digits. 

![LeNet Architecture](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/lenet02.png)

The second architecture is [Traffic Sign Recognition with MultiScale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun. The layers of the CNN are setup like as following:

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

![LeNet02 Architecture](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/lenet22.png) 
  
### Training and testing the data

The classifier is trained over augmented and normalized training dataset. After testing the accuracy over the test dataset (touched at end after desired validation accuracy was achivied), the new images were tested. The images with the highest prediction probabilities from the dataset are viewed against each test image.

![Test Images from internet](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/top_k.png)

![Softmax Probabilities for test output](http://localhost:8888/view/Documents/UD/CarND-Traffic-Sign-Classifier-Project-master/softmax_p.png)

There are still a lot of things we can experiment with such as network architechtures, image preprocessing and never ending fiddling with the hypermeters. 





