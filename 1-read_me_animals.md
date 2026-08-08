# A self-organizing map (SOM) neural network example - clustering animals based on their attributes

This is a read-me file for "animals_code.py" and "animals.txt".

1. **Project description**

The project was focused on using a SOM neural network for automated organization of clusters of animals, based on the attributes they have. The SOM neural network transforms multidimensional data into a two-dimensional output space with unsupervised learning. SOM neural network is initialized with random values around zero, where one "neuron" is a vector with the same length as the vector of one training sample. In this case, the length of the vector was the number of attributes of animals that we considered. We start the training of the SOM with finding the best matching unit (which is one "neuron") for one training example and updating the weights of that unit. We then also change the vectors of neurons/units that are close to the best matching one, where the neighbourhood of which we want to change is up to our decision.

Our input data here are animals and the information of attributes they have, which is coded with zeros and ones, zero meaning that the animal doesn't have that attribute and one that it does.

2. **Usage**

You can run these scripts with a Python interpreter, such as PyCharm, Spyder, Visual Studio Code etc. or you can copy this code to a Jupyter Notebook.

First, make sure that the script "animals_code.py" is in the same directory as the text file with animals' information "animals.txt".

The python script contains 9 different functions:
- find_BMU
- update_weights
- train_SOM
- labels
- table_of_labels
- activity
- activity_vec
- activity_output
- visualize_activity

The first three are used for training the SOM, lables and table_of_labels are used to get the information on which animal does one specific neuron mostly correlate to and the last four are used to get the information on which neurons correlate with one specific animal and how much, where the last one is used for visualization of this activity in a grayscale.

All of these functions are explained in more detail in the third section.

Before using the functions, you need to initialize the SOM by specifying the SOM dimensions, ascribe random values to the SOM neurons and read the information of animal names and their attributes from the text files. An example of this is already done in the script.

After this, you first call the function train_SOM, with the initialized SOM with random values, the training data (animals' attributes) and all the other arguments needed for this function. The train_SOM then uses find_BMU and update_weights to properly train the SOM. After the training is finished, you can use the labels and table_of_labels functions to visualize which training example one neuron best matches to and for this, you put the trained SOM as an argument. The activity functions can be used to inspect the trained SOM and its weights for how much they correlate to one specific training example and visualize_activity takes this activity output as an input, to visualize it.

3. **Description of functions**

(note: all the values that are pre-set don't need to be specified when calling the function, we specify them if we want to call it with a different value than the one that is preset)

	a. find_BMU(SOM, x, act=None):
- SOM is the self-organized map neural network
- x is a vector, in this case one training example (attributes of one animal)
- act is the information of the activity/importance for weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector

> The function computes the euclidean distance of all the neurons in the SOM neural network and returns the one which is most similar to the training example.

> The functions returns the g and h coordinates of the best matching unit in the SOM neural network, with the euclidean distance of this unit to the training example.

	b. update_weights(SOM, train_ex, learn_rate, radius_sq, BMU_coord, step=5):
- SOM is the self-organized map neural network
- train_ex is a vector, which is the current training example
- learn_rate is the learning rate for the SOM, which controls the rate of change of the weight vectors, its value should be between 0 and 1 and it should decrease after each iteration (the value is changed in the train_SOM function)
- radius_sq is the radius (width) square for the neighbourhood and this neighbourhood function decreases after each iteration (the value is changed in the train_SOM function)
- BMU_coord are the best matching unit coordinates (which we get from the find_BMU function)
- step is the size of the neighbourhood we take in consideration for changing the weights, which is predefined as 5, but its size should be changed with respect to the size of the SOM

> The function updates the weights of units of SOM around the best matching one.

> The function returns the SOM with updated weights.

	c. train_SOM(SOM, train_data, act = None, learn_rate=0.9, radius_sq=30, lr_decay=.1, radius_decay=.1, epochs=100):
- SOM is the self-organized map neural network (at first use initialized with random vectors)
- train_data contains all the examples of the training set (in this case all the attributes of the animals)
- act is the information of the activity/importance for the weights in the vector (this will be needed in the find_BMU function and all the weights will be set to one if we don't provide this information when calling the function)
- learn_rate is the learning rate for the SOM, which controls the rate of change of the weight vectors, its value should be between 0 and 1 (in this case it is preset to 0.9, which can be changed when calling the function) and it should decrease after each iteration
- radius_sq is the radius (width) square for the neighbourhood (in this case it is preset to 30, which can be changed when calling the function) and this neighbourhood function decreases after each iteration
- lr_decay is the decay constant for the learning rate, which helps decrease the value of the learning rate after each iteration and is preset to 0.1
- radius_decay is the decay constant for the radius square, which helps decrease the value of the radius after each iteration and is preset to 0.1
- epochs is the number of iterations for the training of the SOM, where in each iteration we go through every training example. This value is preset to 100

> The function traines the SOM in the way that in each iteration it goes through each training example, finds the best matching unit for that example and changes the weights of this unit and the weights of the unit in the near neighbourhood.

> The function returns the trained SOM.

	d. labels(SOM, train_data, lbl):
- SOM is the trained self-organized map neural network
- train_data contains all the examples of the training set (in this case all the attributes of the animals)
- lbl contains an array with data labels (in this case the names of the animals)

> For every neuron in the SOM neural network the function finds which animal it mostly represents.
> The function returns a 2-dimensional array with corresponding labels for every unit ("neuron") of the SOM neural network.

	e. table_of_labels(label, SOM):
- label is the 2-dimensional array of labels, which we get with the labels function
- SOM is the trained self-organized map neural network

> When we call this function, it returns a table with labels of the SOM (where every "neuron" corresponds to one label), but the output of this function can't be saved.

	f. activity(label, names, SOM, train_data, act = None, act_norm = 0, sens = 1):
- label is one label of the training data (in this case one animal (e.g., "cat"))
- names are all the labels of the training data (in this case all the animals)
- SOM is the trained self-organized map neural network
- train_data contains all the examples of the training set (in this case all the attributes of the animals)
- act is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- act_norm is the parameter with which we specify if we want normalized or unnormalized activity, if we don't want normalized activity, we call this function with act_norm = 0, but if we want normalized activity, we call it with act_norm = 1
- sens is the sensitivity with which we convert euclidean distance to activities between 0 and 1, the higher the sensitivity, the lower will be the conversion of the distance (which means that the distance would need to be higher in order to be more "active" - to have the activation number close to one)

> This function inspects every unit of the SOM neural network for how much it represents one specific training example.

> Note that this function only works if you call it with the labels that exist in the training set.

> It returns a 2-dimensional array of numbers between zero and one (which are either normalized or unnormalized), where each unit represents the corresponding unit in the trained SOM and the more the number is close to one, the more that unit represents the label of the training example.

	g. activity_vec(vector, SOM, act = None, act_norm = 0, sens = 1):
- vector is the vector of the same length as the vector of the training examples (in this case a vector of zeros and ones for specific animal attributes)
- SOM is the trained self-organized map neural network
- act is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- act_norm is the parameter with which we specify if we want normalized or unnormalized activity, if we don't want normalized activity we call this function with act_norm = 0 and if we want normalized activity we call it with act_norm = 1
- sens is the sensitivity with which we convert euclidean distance to activities between 0 and 1, the higher the sensitivity, the lower will be the conversion of the distance (which means that the distance would need to be higher in order to be more "active" - to have the activation number close to one)

> This function inspects every unit of the SOM neural network for how much it represents the input vector example (which can either be one vector of the training data or any other vector with combination zeros and ones).

> It returns a 2-dimensional array of numbers between zero and one (which are either normalized or unnormalized), where each unit represents the corresponding unit in the trained SOM and the more the number is close to one, the more that unit represents the vector we put as an input.

	h. act_output(vector, SOM, act = None, output = 0):
- vector is the vector of the same length as the vector of the training examples (in this case a vector of zeros and ones for specific animal attributes)
- SOM is the trained self-organized map neural network
- act is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- output is the parameter with which we specify if we want hard or soft output, if we want hard output we call it with output = 0 and if we want soft output we call it with output = 1

> This function returns the vectors of the unit of the trained SOM, either with the weights of the winner unit (which is the hard output) or as an activity-weighted combination of the weights of all the neurons (soft output).

	i. visualize_activity(activity):
- activity is the 2-dimensional array of numbers between zero and one that we get either with the activity or the activity_vec function

> When we call this function, it visualizes the activity numbers of the 2-dimensional array (the SOM) in a grayscale, but the output of this function can't be saved.

Note: the functions in this form can be used for any kind of numerical input data, you just have to change the SOM dimensions and all the parameters accordingly.

4. **Credits and literature**

This project was created as a test example of implementation of a SOM neural network for a project of the master's programme of Cognitive Science at Comenius University in Bratislava, under the supervision of Assoc. Prof. Martin Takáč.

The functions find_BMU, update_weights and train_SOM were copied and changed in order to work in this specific case from Saeed, M.: Self-Organizing Maps: Theory and Implementation in Python with NumPy [https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/](https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/)

You can read more on SOM neural networks in the following papers:
- Asan, U. & Ercan, S. (2012). An Introduction to Self-Organizing Maps. In: Kahraman, C. (ed). *Computational Intelligence Systems in Industrial Engineering: with Recent Theory and Applications* (pp. 299-319). doi: 10.2991/978-94-91216-77-0_14
- Ritter, H. & Kohonen, T. (1989). Self-Organizing Semantic Maps. *Biological Cybernetics, 61*, 241–254. [https://doi.org/10.1007/BF00203171](https://doi.org/10.1007/BF00203171)
