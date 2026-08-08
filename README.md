# Overview of the read-me files for the project "Event sequencing module for a model of early social cognition"

This is a read-me file for the general overview of all the read me files that were made as a part of the project of implementation of the C-block model. The C-block model was implemented with the help of self-organizing map (SOM) neural networks. The SOM neural networks are briefly described in the "read_me_animals.md" file and the C-block model is briefly described in all the other read-me files. All the files also contain a list of literature if you wish to dive deeper into the SOM neural networks and the C-block model.

The read me files are the following:
- *read_me_animals*
- *read_me_cblock_with_sequencer1*
- *read_me_cblock_with_sequencer2*
- *read_me_whole_cblock*
- *read_me_whole_cblock_withIOR*

The order in which they are listed follows the order in which they were made and the order of increasing complexity of the C-block model. All of the files can be read and used independantly, with the exception of the last one, namely the "read_me_whole_cblock_withIOR.md".

All of the read-me files have the same structure: first, the project is briefly described, then the usage of the functions of the script that the file refers to is discussed, then the functions are described in greater detail and lastly, the files contain a credits and literature section.

Note that if you want to use the whole C-block model with both the sequencer and planner SOMs, you can skip all previous files and use just the *read_me_whole_cblock* and *read_me_whole_cblock_withIOR* files. Otherwise you can use all the previous ones in order to understand better how the SOM neural networks and the C-block model work.

Here is a brief description of the contents of the read-me files:

1. **read_me_animals** *(this file can be read and used independently)*

This file contains the information of the test version of SOM we implemented for our project. In our case, we used a list of animals and their attributes as the training data for the SOM. Note that this is not a part of the C-block model.

2. **read_me_cblock_with_sequencer1** *(this file can be read and used independently)*

This file refers to our first implementation of the C-block model with the sequencer SOM. The training od the sequencer SOM relies on a predefined dataset and it is the least complicated version of the sequencer SOM that we implemented. We used a list of names as the training data.

3. **read_me_cblock_with_sequencer2** *(this file can be read and used independently)*

We then improved the sequencer SOM of the C-block model, so that the training set is gradually developing and the SOM learns after each new element of a sequence is presented.

4. **read_me_whole_cblock** *(this file can be read and used independently)*

This is the last and the most complicated version of the C-block that we implemented and the only one that contains both sequencer SOM and planner SOM of the model. After the sequencer SOM is trained, the planner SOM is also trained for the whole sequence.

5. **read_me_whole_cblock_withIOR** *(this file is dependent on the read_me_whole_cblock file)*

This file refers only to one version of testing the C-block model, whilst the training is described in the previous file. The version of testing the C-block model is a bit different and more complex than in the previous file, because it also takes into account the inhibition of return of the predicted plan, so that next time when searching the best plan, it searches for an alternative. The inhibition of return applies to the testing of the planner SOM.
