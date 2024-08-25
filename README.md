# Flower-Image-Classification-using-CNN
This project implements a Convolutional Neural Network (CNN) for image classification, specifically designed to classify images into 17 different categories, corresponding to the classes in the 17 Category Flower Dataset. Below is a detailed breakdown of the model architecture:

The model starts with three convolutional layers with increasing filters (32, 64, 128) to extract features, each followed by a max-pooling layer to reduce spatial dimensions. After flattening, two fully connected layers with ReLU activation and L2 regularization learn complex patterns, while dropout layers help prevent overfitting. Finally, a softmax-activated output layer provides class probabilities. EarlyStopping has been also used to monitor validaiton loss and control the training.

This project utilizes a modified  [17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) provided by the Visual Geometry Group at the University of Oxford.

Their paper can be read here: Nilsback, M. E., & Zisserman, A. (2008). "Automated Flower Classification over a Large Number of Classes." In *Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing*.

The best model I have found is a training accuracy of 71% and a test accuracy of 62 %. There is some overfitting but it could be because I have downsized the dataset. 

# Disclaimer
Note: The code in this repository was developed by me as part of an online course assignment. The dataset used, the 17 Category Flower Dataset, is owned by the Visual Geometry Group at the University of Oxford. All rights to the dataset belong to its original creators. This code is provided for educational purposes and should not be used for commercial purposes without proper permissions from the dataset owners.
