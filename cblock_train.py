import numpy as np

# Dimensions of the SOM grid
m = 50
n = 50
# Initialize the training data
train_data = np.genfromtxt("train_data_mod.txt", delimiter=",", usecols=range(1, 401))
letters = np.genfromtxt("train_data_mod.txt", delimiter=",", usecols=0, dtype="U")
# Initialize the SOM randomly
rand = np.random.RandomState(0)
SOM = np.random.rand(m, n, 400)

# Return the (g,h) index of the BMU in the grid and the euclidean distance
def find_BMU(SOM, x, act=None):
    if np.all(act == None):  # if the attention weights are not specified, this sets them all to ones
        act = np.array([1]*len(SOM[0, 0]))
    distSq = (np.square(np.multiply(act, (SOM - x)))).sum(axis=2)
    g, h = np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)  # the function converts from linear to 2D index
    eucl_dist = distSq[g, h]
    return g, h, eucl_dist

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
def train_SOM(SOM, train_data, act=None, learn_rate=0.9, radius_sq=30,
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

# an example of how to call the functions to train the data
SOM = train_SOM(SOM, train_data, epochs=30)
np.save("trained_SOM_mod.npy", SOM)
