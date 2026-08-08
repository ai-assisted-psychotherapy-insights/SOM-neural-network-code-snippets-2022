# C-block model with the sequencer part and predefined training set

This is a read-me file for "cblock.py", "cblock_mod.py", "cblock_train.py", "cblock_test.py", "cblock_test_mod.py" and "names.txt".

1. **Project description**

The project was focused on implementing a C-block model, which is a connectionist model that can detect sequences in data and represent them as chunks or plans. The C-block model is divided in two parts, here we implemented only the sequencer part, which is able to predict the following element in a sequence and learn declarative representations of sequences. The model works on the system of self-organizing map (SOM) neural network. It is structured into four parts: next, recent, context and tonic. Next contains the information of the current unit of the sequence, recent contains the information of the previous unit, context of all the previous units, where the most recent is the strongest one and tonic contains the information of the whole sequence, where (just the opposite as context) the first unit is the strongest one.

The input data for the system in this case are names as they are written in the "names.txt" file.

2. **Usage**

You can run these scripts with a Python interpreter, such as PyCharm, Spyder, Visual Studio Code etc. or you can copy this code to a Jupyter Notebook.

Firstly, make sure that all of the listed files are in the same working directory. There are two alternative ways of usage, the difference being the information storage in the tonic part of the sequencer. If you want one unit of the whole tonic part to represent the whole sequence, you can use the combination of files "cblock.py", "cblock_train.py" and "cblock_test.py".

If you want your tonic to gradually change while new information of the sequence is being presented, use the combination of files "cblock_mod.py", "cblock_train.py" and "cblock_test_mod.py". In both cases you first have to get the training data with either running "cblock.py" or "cblock_mod.py". When you get the training data, you run "cblock_train.py" and after this you can test the model with running "cblock_test.py" or "cblock_test_mod.py", accordingly.

The "cblock.py" and "cblock_mod.py" contain one function (here you first need to read the text file):
- data (which is a bit different amongst the scripts)

The "cblock_train.py" contains three functions (here, you first need to specify the dimensions of SOM and read the training data):
- find_BMU
- update_weights
- train_SOM

The "cblock_test.py" contains four functions and "cblock_test_mod.py" contains one more (in both cases you first need to read the training data file):
- find_BMU
- activity_vec
- act_output
- test
- tonics (this one is present just in "cblock_test_mod.py")

First, run the data function to get the training data in a proper format (note that to run this function, you have to have the training examples in a seperate text file). Then save this training data to a numpy file and run the train_SOM function in the separate script, with first specifying the SOM dimensions and initializing the vectors of the neurons to random values. The train_SOM uses find_BMU and update_weights to train the data. After the data is trained, save it to a numpy file and test the trained SOM separately in the third script. Use the trained SOM as an argument in the test function and if you are using the script with the gradually changing tonic, you first need to run the tonics function to get the information of the tonic values and then you can use this in the test function. The functions find_BMU, activity_vec and act_output are called by the test function.

3. **Description of functions**

(note: all the values that are pre-set don't need to be specified when calling the function, we specify them if we want to call it with a different value than the one that is preset)

	a.1. [in "cblock.py"] data(names, const = 0.9): 
- *names* is the data from which we want to generate the data set, in this case a list of names
- *const* is the constant with which the information in the context decreases after a new unit is presented and it is preset to 0.9

> The function creates the data set, where each unit of a sequence is presented as a 400-dimensional vector, the first 100 representing the tonic, the second 100 the context, the third 100 the recent and the last 100 the next part.

> In this case the tonic value is the same for every unit of a sequence.

> The function returns the data set for all the units (in this case all the letters of the names, including the end of word), if you want to use it for training, save it to a numpy file.

	a.2. [in "cblock_mod.py"] data(names, decay1 = 0.9, decay2 = 0.7, m = 1):
- *names* is the data from which we want to generate the data set, in this case a list of names
- *decay1* is the constant with which the information in the context decreases after a new unit is presented and it is preset to 0.9
- *decay2* is the constant with which the importance of every new unit of the sequence is decreased and it is preset to 0.7
- *m* is the constant with which we decrease the importance of the tonic after a new unit is presented and its value is decreased after every iteration with the constant decay2. It is preset to the value of 1

> The function creates the data set, where each unit of a sequence is presented as a 400-values vector, the first 100 representing the tonic, the second 100 the context, the third 100 the recent and the last 100 the next part.

> In this case the tonic value changes after a new unit of the sequence is presented.

> The function returns the data set for all the units (in this case all the letters of the names, including the end of word), if you want to use it for training, save it to a numpy file.

	b. [in "cblock_train.py", "cblock_test.py" and "cblock_test_mod.py"] find_BMU(SOM, x, act=None):
- *SOM* is the self-organized map neural network
- *x* is a vector, in this case one training example (the 400-values vector of one unit)
- *act* is the information of the activity/importance for weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector

> The function computes the euclidean distance of all the neurons in the SOM neural network and returns the one which is most similar to the training example.

> The functions returns the g and h coordinates of the best matching unit in the SOM neural network, with the euclidean distance of this unit to the training example.

	c. [in "cblock_train.py"] update_weights(SOM, train_ex, learn_rate, radius_sq, BMU_coord, step=5):
- *SOM* is the self-organized map neural network
- *train_ex* is a vector, which is the current training example
- *learn_rate* is the learning rate for the SOM, which controls the rate of change of the weight vectors, its value should be between 0 and 1 and it should decrease after each iteration (the value is changed in the train_SOM function)
- *radius_sq* is the radius (width) square for the neighbourhood and this neighbourhood function decreases after each iteration (the value is changed in the train_SOM function)
- *BMU_coord* are the best matching unit coordinates (which we get from the find_BMU function)
- *step* is the size of the neighbourhood we take in consideration for changing the weights, which is predefined as 5, but its size should be changed with respect to the size of the SOM

> The function updates the weights of units of SOM around the best matching one.

> The function returns the SOM with updated weights.

	d. [in "cblock_train.py"] train_SOM(SOM, train_data, act = None, learn_rate=0.9, radius_sq=30, lr_decay=.1, radius_decay=.1, epochs=100):
- *SOM* is the self-organized map neural network (at first use initialized with random vectors)
- *train_data* contains all the examples of the training set (in this case the 400-values vector of all the units)
- *act* is the information of the activity/importance for the weights in the vector (this will be needed in the find_BMU function and all the weights will be set to one if we don't provide this information when calling the function)
- *learn_rate* is the learning rate for the SOM, which controls the rate of change of the weight vectors, its value should be between 0 and 1 (in this case it is preset to 0.9, which can be changed when calling the function) and it should decrease after each iteration
- *radius_sq* is the radius (width) square for the neighbourhood (in this case it is preset to 30, which can be changed when calling the function) and this neighbourhood function decreases after each iteration
- *lr_decay* is the decay constant for the learning rate, which helps decrease the value of the learning rate after each iteration and is preset to 0.1
- *radius_decay* is the decay constant for the radius square, which helps decrease the value of the radius after each iteration and is preset to 0.1
- *epochs* is the number of iterations for the training of the SOM, where in each iteration we go through every training example. This value is preset to 100

> The function traines the SOM in the way that in each iteration it goes through each training example, finds the best matching unit for that example and changes the weights of this unit and the weights of the unit in the near neighbourhood.

> The function returns the trained SOM.

	e. [in "cblock_test.py" and "cblock_test_mod.py"] activity_vec(vector, SOM, act = None, act_norm = 0, sens = 1):
- *vector* is the vector of the same length as the vector of the training examples (in this case a 400-values vector)
- *SOM* is the trained self-organized map neural network
- *act* is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- *act_norm* is the parameter with which we specify if we want normalized or unnormalized activity, if we don't want normalized activity we call this function with act_norm = 0 and if we want normalized activity we call it with act_norm = 1
- *sens* is the sensitivity with which we convert euclidean distance to activities between 0 and 1, the higher the sensitivity, the lower will be the conversion of the distance (which means that the distance would need to be higher in order to be more "active" - to have the activation number close to one)

> This function inspects every unit of the SOM neural network for how much it represents the input vector example (which can either be one vector of the training data or any other vector with the same length).

> It returns a 2-dimensional array of numbers between zero and one (which are either normalized or unnormalized), where each unit represents the corresponding unit in the trained SOM and the more the number is close to one, the more that unit represents the vector we put as an input.

	f. [in "cblock_test.py" and "cblock_test_mod.py"] act_output(vector, SOM, act = None, output = 0, sens = 1):
- *vector* is the vector of the same length as the vector of the training examples (in this case a 400-values vector)
- *SOM* is the trained self-organized map neural network
- *act* is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, otherwise we provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- *output* is the parameter with which we specify if we want hard or soft output, if we want hard output we call it with output = 0 and if we want soft output we call it with output = 1
- *sens* is the sensitivity with which we convert euclidean distance to activities between 0 and 1 and it is used when calling the function activity_vec within this function

> This function returns the vectors of the unit of the trained SOM, which is the best matching one to the input vector. It returns either the weights of the winner unit (which is the hard output) or as an activity-weighted combination of the weights of all the neurons (soft output).

	g.1. [in "cblock_test.py"] test(tonic_no, SOM, act=None, output=0, sens = 1, decay = 0.9):
- *tonic_no* is the index, where the value of the tonic is 1 (within - in this case - the 100-values long vector of tonic)
- *SOM* is the trained self-organized map neural network
- *act* is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, unless for the values of the next part, which will be all zeros - otherwise we need to provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- *output* is the parameter with which we specify if we want hard or soft output, which we need for calling the function act_output
- *sens* is the sensitivity with which we convert euclidean distance to activities between 0 and 1, which we need for calling the function act_output
- *decay* is the decay constant for the context values

> This function predicts the sequence based on the tonic index and the trained SOM.

> Note that this function only works if one sequence in your training data has always the same tonic, which is a value of one amongst all zeros.

> This function returns the predicted sequence (in this case a name).

	g.2. [in "cblock_test_mod.py"] test(tonic_vec, SOM, act=None, output=0, sens = 1, decay = 0.9):
- *tonic_vec* is the vector tonic of a sequence (in this case 100-values long)
- *SOM* is the trained self-organized map neural network
- *act* is the information of the activity/importance for the weights in the vector - if we don't provide this information, all of the weights will have the same importance, unless for the values of the next part, which will be all zeros - otherwise we need to provide a vector with weight values, with which we put more or less importance to specific weights of the vector
- *output* is the parameter with which we specify if we want hard or soft output, which we need for calling the function act_output
- *sens* is the sensitivity with which we convert euclidean distance to activities between 0 and 1, which we need for calling the function act_output
- *decay* is the decay constant for the context values

> This function predicts the sequence based on the tonic vector and the trained SOM.

> Note that this function only works in the version where the tonic changes after every new unit of a sequence, and when using this function, we take the tonic that was generated at the end of the sequence.

> This function returns the predicted sequence (in this case a name).

	h. [in "cblock_test_mod.py"] tonics(letters, train_data):
- *letters* is a list with all the letters of the training set
- *train_data* contains all the 400-values long vectors of the training set

> This function finds and returns the tonics of all the sequences (the tonics that were generated at the end of a sequence).
> Note that this function only works in the version where the tonic changes after every new unit of a sequence.

4. **Credits and literature**

This project was created as a first attempt on the implementation of the C-block model for a project of the master's programme of Cognitive Science at Comenius University in Bratislava, under the supervision of Assoc. Prof. Martin Takáč.

The functions find_BMU, update_weights and train_SOM were copied and changed in order to work in this specific case from Saeed, M.: Self-Organizing Maps: Theory and Implementation in Python with NumPy [https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/](https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/)

You can read more on the C-block in the following papers:
- Takáč, M., Knott, A. & Sagar, M. (2020a). *C-block: A system for learning motor plans with perceptual consequences*. 1st SMILES (Sensorimotor Interaction, Language and Embodiment of Symbols) workshop, ICDL 2020, Nov 2020, Valparaiso, Chile. hal-02985188
- Takáč, M., Knott, A. & Sagar, M. (2020b). SOM-Based System for Sequence Chunking and Planning. In: Farkaš I., Masulli P., Wermter S. (eds). *Artificial Neural Networks and Machine Learning – ICANN 2020.* ICANN 2020. Lecture Notes in Computer Science, vol 12396. Springer, Cham.	[https://doi.org/10.1007/978-3-030-61609-0_53](https://doi.org/10.1007/978-3-030-61609-0_53)

You can read more on SOM neural networks in the following papers:
- Asan, U. & Ercan, S. (2012). An Introduction to Self-Organizing Maps. In: Kahraman, C. (ed). *Computational Intelligence Systems in Industrial Engineering: with Recent Theory and Applications* (pp. 299-319). doi: 10.2991/978-94-91216-77-0_14
- Ritter, H. & Kohonen, T. (1989). Self-Organizing Semantic Maps. *Biological Cybernetics, 61*, 241–254. [https://doi.org/10.1007/BF00203171](https://doi.org/10.1007/BF00203171)
