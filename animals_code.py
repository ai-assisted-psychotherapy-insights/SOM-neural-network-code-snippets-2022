import numpy as np

# Dimensions of the SOM grid
m = 20
n = 20
# Number of training examples
n_x = 6000
# Initialize the training data
train_data = np.genfromtxt("animals.txt", delimiter=",", skip_header=True, usecols=range(1, 14))
animals = np.genfromtxt("animals.txt", delimiter=",", skip_header=True, usecols=0, dtype="U")
attributes = np.genfromtxt("animals.txt", delimiter=",", usecols=range(1, 13), dtype="U", skip_footer=16)
# Initialize the SOM randomly
rand = np.random.RandomState(0)
SOM = np.random.rand(20, 20, 13)

# Return the (g,h) index of the BMU in the grid
def find_BMU(SOM, x, act=None):
    if np.all(act == None):  # if the attention weights are not specified, this sets them all to ones
        act = np.array([1]*len(SOM[0, 0]))
    distSq = (np.square(np.multiply(act, (SOM - x)))).sum(axis=2)
    g, h = np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)  # the function converts from linear to 2D index
    eucl_dist = distSq[g, h]
    return g, h, eucl_dist
"""
# Update the weights of the SOM cells when given a single training example
# and the model parameters along with BMU coordinates as a tuple
def update_weights(SOM, train_ex, learn_rate, radius_sq,
                   BMU_coord, step=5):
    g, h = BMU_coord
    # if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g, h, :] += learn_rate * (train_ex - SOM[g, h, :])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g - step), min(SOM.shape[0], g + step)):
        for j in range(max(0, h - step), min(SOM.shape[1], h + step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / (2 * radius_sq))
            SOM[i, j, :] += learn_rate * dist_func * (train_ex - SOM[i, j, :])
    return SOM

# Main routine for training an SOM. It requires an initialized SOM grid
# or a partially trained grid as parameter
def train_SOM(SOM, train_data, act=None, learn_rate=0.9, radius_sq=25,
              lr_decay=.1, radius_decay=.1, epochs=100):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)
        for train_ex, i in zip(train_data, range(len(train_data))):
            g, h, ed = find_BMU(SOM, train_ex, act)
            SOM = update_weights(SOM, train_ex,
                                 learn_rate, radius_sq, (g, h))
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)
    return SOM
"""
def labels(SOM, train_data, lbl):
    label = np.array([["       "] * len(SOM[0])] * len(SOM[1]))
    for x in range(len(SOM[0])):
        for y in range(len(SOM[1])):
            i = 0
            mindist = float("inf")
            for train_ex in train_data:
                d = np.sum(np.square(SOM[x, y] - train_ex))
                if d < mindist:
                    mindist = d
                    label[x, y] = lbl[i]
                i += 1
    return label
"""
# an example of how to train the data
a = np.array([1]*13)
a[4] = 0
SOM = train_SOM(SOM, train_data, a, epochs=50)
"""

# how to save the data in a numpy file so that you can use the same trained SOM
"""np.save("a3_SOM.npy", SOM)"""
SOM = np.load("a2_SOM.npy")

an_label = labels(SOM, train_data, animals)
# print(an_label)

# an example of how to save the data to a text file
"""np.savetxt("file2.txt", an_label, fmt="%s", delimiter=",")"""

# this prints a table of labels (we can't save the table, it just prints, when calling the function)
def table_of_labels(label, SOM):
    nr = np.array(range(1, len(SOM[0]+1)))
    row_format ="{:>8}" * (len(nr))
    print(row_format.format("", *nr))
    for n_, row in zip(nr, label):
        print(row_format.format(n_, *row))

# table_of_labels(an_label,SOM)

def activity(label, names, SOM, train_data, act=None, act_norm=0, sens=1):
    if np.all(act == None):  # if this is not specified, the attention weights are all set to ones
        act = np.array([1]*len(SOM[0, 0]))
    i = np.where(names == label)
    dist = (np.square(np.multiply(act, (SOM - train_data[i])))).sum(axis=2)
    activity = np.exp(-dist * sens)
    if act_norm == 0:  # if we don't want normalized activity
        return activity
    elif act_norm == 1:  # if we want normalized activity
        activity_n = activity/np.sum(activity)
        return activity_n

def activity_vec(vector, SOM, act=None, act_norm=0, sens=1):
    if np.all(act == None):  # if this is not specified, the attention weights are all set to ones
        act = np.array([1]*len(SOM[0, 0]))
    dist = (np.square(np.multiply(act, (SOM - vector)))).sum(axis=2)
    activity = np.exp(-dist * sens)
    if act_norm == 0:  # if we don't want normalized activity
        return activity
    elif act_norm == 1:  # if we want normalized activity
        activity_n = activity/np.sum(activity)
        return activity_n

def act_output(vector, SOM, act=None, output=0):
    if np.all(act == None):  # if this is not specified, the attention weights are all set to ones
        act = np.array([1]*len(SOM[0, 0]))
    if output == 0:  # if we want "hard" output
        g, h, eucl_dist = find_BMU(SOM, vector, act)
        return SOM[g, h]
    elif output == 1:  # if we want "soft" output
        activity_n = activity_vec(vector, SOM, act, act_norm=1)
        sm = np.zeros(len(SOM[0, 0]))
        for i, row in enumerate(activity_n):
            for j, neuron in enumerate(row):
                sm += np.multiply(neuron, SOM[i, j])
        return sm

# an example of how to compute the activities and visualize it
a = activity("zebra", animals, SOM, train_data)
#print(a)

b = activity_vec(vector=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), SOM=SOM, act_norm=1)
#print(b)

c = act_output(vector=np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]), SOM=SOM, output=1)
#print(c)

# to visualize the activity of data
import matplotlib.pyplot as plt

# we can't save the picture, it just displays when we call the function
def visualize_activity(activity):
    plt.imshow(activity, cmap='gray')
    plt.show()

visualize_activity(a)
