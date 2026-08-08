import numpy as np

# Return the (g,h) index of the BMU in the grid
def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    g,h = np.unravel_index(np.argmin(distSq, axis=None), distSq.shape) # the function converts from linear to 2D index
    eucl_dist = distSq[g,h]
    # print(eucl_dist)
    # eucl_dist2 = np.min(distSq)
    # print(eucl_dist2)
    return [g,h,eucl_dist]


# Update the weights of the SOM cells when given a single training example
# and the model parameters along with BMU coordinates as a tuple
def update_weights(SOM, train_ex, learn_rate, radius_sq,
                   BMU_coord):
    g, h = BMU_coord
    # if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g, h, :] += learn_rate * (train_ex - SOM[g, h, :])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(0, SOM.shape[0]):
        for j in range(0, SOM.shape[1]):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / (2 * radius_sq))
            # dist_func = np.exp(-dist_sq / radius_sq)
            # print(dist_func)
            SOM[i, j, :] += learn_rate * dist_func * (train_ex - SOM[i, j, :])
    return SOM


# Main routine for training an SOM. It requires an initialized SOM grid
# or a partially trained grid as parameter
def train_SOM(SOM, train_data, learn_rate=0.9, radius_sq=30,
              lr_decay=.1, radius_decay=.1, epochs=100):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)
        qe=0
        for train_ex, i in zip(train_data,range(len(train_data))):
            g, h, ed = find_BMU(SOM, train_ex)
            qe+=ed
            SOM = update_weights(SOM, train_ex,
                                 learn_rate, radius_sq, (g, h))
        # print(qe/len(train_data))
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        # print("L_R=",learn_rate,"epoch=",epoch)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        # print("R_S=",radius_sq,"epoch=",epoch)
    return SOM

# Dimensions of the SOM grid
m = 20
n = 20
# Number of training examples
n_x = 6000
rand = np.random.RandomState(0)
# Initialize the training data
train_data = np.genfromtxt("animals.txt", delimiter=",",skip_header=True,usecols=range(1,14))
animals = np.genfromtxt("animals.txt", delimiter=",",skip_header=True,usecols=0,dtype="U")
attributes = np.genfromtxt("animals.txt", delimiter=",",usecols=range(1,13),dtype="U",skip_footer=16)
# Initialize the SOM randomly
#SOM = rand.randint(0, 1, (m, n, 13)).astype(float) # 13 attributes
SOM = np.random.rand(20,20,13)


SOM = train_SOM(SOM, train_data, epochs=50)

def labels(SOM, train_data, lbl):
    label = np.array([["       "] * len(SOM[0])] * len(SOM[1]))
    # activity = np.array([[0] * 20] * 20).astype(float)
    # eucld = np.array([[0] * 20] * 20).astype(float)
    for x in range(len(SOM[0])):
        for y in range(len(SOM[1])):
            i = 0
            mindist = float("inf")
            for train_ex in train_data:
                d = np.sum(np.square(SOM[x, y] - train_ex))
                # a = np.exp2(d)
                if d < mindist:
                    mindist = d
                    label[x, y] = lbl[i]
                    # activity[x, y] = a
                    # eucld[x, y] = d
                i += 1
    return label

an_label = labels(SOM, train_data, animals)
# print(an_label)

n = np.array(range(1,21))
row_format ="{:>10}" * (len(n))
print(row_format.format("", *n))
for n_, row in zip(n,an_label):
    print(row_format.format(n_,*row))

def activity(label, names, SOM, train_data):
    i = np.where(names == label)
    dist = (np.square(SOM - train_data[i])).sum(axis=2)
    act = np.exp(-dist)
    return act

a = activity("cat", animals, SOM, train_data)
# print(a)


# to visualize the activity of data
import matplotlib.pyplot as plt

plt.imshow(a, cmap='gray')
plt.show()