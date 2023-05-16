UW-EE399-Assignment-5
=========
This holds the code and backing for the fifth assignment of the EE399 class. T

Project Author: Elijah Reeb, elireeb@uw.edu

.. contents:: Table of Contents

Homework 5
---------------------
Introduction
^^^^^^^^^^^^


Theoretical Backgroud
^^^^^^^^^^^^

.. image:: ![Uploading image.png…]()


Algorithm Implementation and Development
^^^^^^^^^^^^
With the ease of pytorch packages, FFNN code is simple to develop. After importing the necessary tools 4 main functions are called in order to setup train and test a neural network. They are broken up below. The first block involves setting the parameters of the layers and their size. For part I of this assignment a simple neural net is used with a 1 dimentional input and output. Those are the first ones. And the middle layers of 20 and 10 are set up next. These values are again from hyper-parameter tuning, but can be a large range. 

.. code-block:: text



Computational Results
^^^^^^^^^^^^

Summary and Conclusions
^^^^^^^^^^^^
To conclude part I, it is clear that the rule of needing a large dataset in order to use a neural network is very true. Twenty training points are not close to enough to determine a model and that is shown by the model not even being able to determine a simple task of finding a slope of some points. It appears that the model just determined the average of the points and set the input to result in that output. This model is expensive computationally as it runs 1000 epochs just to figure out nothing. This shows a good illustration of a case when not to use a NN. 

Switching to part II, The four models have comparable success on the testing data. With a large enough test set of 56000 images it is clear that a neural net is able to reach a reasonable amount of accuracy. The four models took comparable amounts of time to run so there was not as much trade off between computation power (not exactly measured) and accuracy. Although, processing the data into the 20 PCA components was helpful in reducing the amount of time each model took. We can see that the LSTM model was the most effective and this is likely because of how it stores the data. This was the most difficult model to implement and its complexity will be explained in a later assignment. The SVM method was second most effective which shows how more simple linear algebra helps an issue like this. Another main difference between the models is their handling of a more robust dataset. SVM may be scaled to look at higher dimensions however if the data is translated or has its size changed in the input images then there will be added difficulty and the SVM and decision tree methods as they do not scale as well where linear algebra is not as effective. However, there is a large area of change avaiable for FFNN and the LSTM models. The potential to add a convolution window as well as pooling layers (as well as just more layers with different functions in general) may allow for these models to perform on a different set of digits with less similar features in general. 
