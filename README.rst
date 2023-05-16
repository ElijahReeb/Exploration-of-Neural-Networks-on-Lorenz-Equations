UW-EE399-Assignment-5
=========
This holds the code and backing for the fifth assignment of the EE399 class. This assignment involves the generation of seemingly "random" data using the Lorenz equations. This data is then input into 4 different neural network models in order to demonstrate their ability to advance the solution in time as well as future state prediction for data generated under different parameter values in the equations. 

Project Author: Elijah Reeb, elireeb@uw.edu

.. contents:: Table of Contents

Homework 5
---------------------
Introduction
^^^^^^^^^^^^
This is the most complext assignment to date. Due to time constraints not all of the proper backing was found or implemented in the code. This assignment called for setting up the Lorenz equations and manipulation of the rho parameter. The training data were the inputs of rho = 10,28, and 40. Unlike previous assignments where an input data point had a specific output such as an MNIST image being mapped to a digit, this assignment involves mapping the value of the equation at time t to the value at t + delta t. This training data was set up and combined to be used to create fits for the neural networks and then the fits were applied to rho values of 17 and 35. The four models used were the Feed Forward Neural Net (FFNN), the Long Short-Term Memory model (LSTM), a Recurrent Neural Network (RNN) and an Echo State Network (ESN). These models use varying algorithms and layer structures in order to process the input information and generate an output. 

Theoretical Backgroud
^^^^^^^^^^^^
Briefly, we will look into the background behind the Lorenz equations. For the generation of these examples, beta and sigma are held at 8/3 and 10 respectively. We can see a difference in the graphs below that when rho > 27 (or beta * sigma) there is a difference in the level of how much the graph moves from one "disc" to the other. The lorenz equations are implemented in the code below. Each value changes as a result of the previous time steps so we should expect a pattern that could potentially be found by the neural network models. 

.. code-block:: text

         def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
          x, y, z = x_y_z
          return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

Below are the graphical representations of the training data. The data were combined into one array in order to allow for faster fitting of the neural networks.

.. image:: https://github.com/ElijahReeb/UW-EE399-Assignment-5/assets/130190276/797a26fe-a126-40dc-a135-5e68023a27ee


.. image:: https://github.com/ElijahReeb/UW-EE399-Assignment-5/assets/130190276/e5539899-01c5-45de-90db-abc1cf7d96a4

With the goal of creating the following outputs as the specific test data. When rho is set to 17 and 35. 

.. image:: https://github.com/ElijahReeb/UW-EE399-Assignment-5/assets/130190276/167dbc34-e08f-4b9a-83c4-f14e345c1755

Transitioning to some of the theoretical background behind the Network models. As discussed in assignment 4 a neural network involves layers connected by weights and activation functions. They are trained through backpropagation where the weights are adjusted based on an error metric. The LSTM is a type of RNN which means the network is more cyclic meaning nodes may be able to input to themselves. The ESN is also a type of RNN which has a random hidden layer. These networks are more designed for time series data so we would expect at least one of these algorithms to out perform the FFNN. This will be discussed later.

Algorithm Implementation and Development
^^^^^^^^^^^^
Similar to previous assignments, Python packages such as pytorch allow very easy implementation of all algorithms. One just has to import the package, find some starter code to help with the syntax of creating a frame work and then run a few commands to fit their data and test. The next section shows some of the coding used to set up the neural network frameworks to show on the surface the models are vary similar. 

Below is the framework to create the FFNN. This has the input and output layers as well as 3 hidden layers which can be hyper parameter tuned to hope for the most sucess in testing. 

.. code-block:: text

        self.fc1 = nn.Linear(in_features=3, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=30)
        self.fc3 = nn.Linear(in_features=30, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=3)

Next we can observe the backing behind the LSTM model. Intital parameters of the input and output size are fed in (3 in our case) as well as the hidden size and number of layers. LSTMs also implement batches in order to construct how training will be done. 

.. code-block:: text

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
 
 Next in the RNN model. We see that similar to the LSTM the input an output size variables are important to the framework. In this neural network structure we see it is still very similar in overall frame work to the models above. 
 
.. code-block:: text

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

Last the Echo State model. This model has the similar setup to the others where layers are defined. However it greatly differs in the commands highlighed below which show how the weights are more observed and can be set by the user as well as the implementing of a mask. This potentially allows the user to change more parameters without going into the backend.

.. code-block:: text

        mask = torch.rand_like(self.input_weights)
        mask = mask < sparsity
        self.input_weights *= mask.float()
        self.hidden_weights *= spectral_radius /       torch.max(torch.abs(torch.linalg.eig(self.hidden_weights)[0]))

To summarize the above, we can observe that the algorithms that a user is implementing in the close side (me) are all very similar. Most users do not interact with the backend code at all. This means it is cruical to compare models and test models against each other model to model because one is not entirely able to change the whole algorithm without shifting to a different model. 

Computational Results
^^^^^^^^^^^^
Due to difficulty in code it was hard to get full compuational results from each of the models. The FFNN loss graph is shown below. We see that the model gets slightly better as the loss continues to decrease. With different parameters set one could observe better or worse loss functions. When this model was applied to the data it did not do a good job replicating the test data. This has to do with how the lorenz equations change. The model was very ineffective when predicting rho = 35 but had much less error in predicting rho = 17. 

.. image:: https://github.com/ElijahReeb/UW-EE399-Assignment-5/assets/130190276/fcfec198-2fc4-4256-9e13-33c1b4e4b1e2

Summary and Conclusions
^^^^^^^^^^^
Upon discussion with a peer that had more coding success, the graph below was attained. As expected a RNN model is more effective in predicting based on the training data. These graphs may not be completely accurate as different levels of hyperparameter tuning were done. One model with different parameters may do better than the others. In this case we obesrve the ESN has a much higher error compared to the rest. There were different errors when looking at rho = 17 vs rho = 35 as the predictibilty changed based off of the Lorenz equations.  

.. image:: https://github.com/ElijahReeb/UW-EE399-Assignment-5/assets/130190276/0a0f713d-75e8-4338-9b44-98afe25b6a38
