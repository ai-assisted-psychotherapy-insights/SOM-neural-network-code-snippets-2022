import numpy as np

# load the trained SOM
SOM = np.load("trained_SOM.npy")

def find_BMU(SOM, x, act=None):
    if np.all(act == None):  # if the attention weights are not specified, this sets them all to ones
        act = np.array([1]*len(SOM[0, 0]))
    distSq = (np.square(np.multiply(act, (SOM - x)))).sum(axis=2)
    g, h = np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)  # the function converts from linear to 2D index
    eucl_dist = distSq[g, h]
    return g, h, eucl_dist

def activity_vec(vector, SOM, act=None, act_norm=0, sens=1):
    if np.all(act == None):  # if the attention weights are not specified, this sets them all to ones
        act = np.array([1]*len(SOM[0, 0]))
    dist = (np.square(np.multiply(act, (SOM - vector)))).sum(axis=2)
    activity = np.exp(-dist * sens)
    if act_norm == 0:  # if we don't want normalized activity
        return activity
    elif act_norm == 1:  # if we want normalized activity
        activity_n = activity/np.sum(activity)
        return activity_n

def act_output(vector, SOM, act=None, output=0, sens=1):
    if np.all(act == None):  # if the attention weights are not specified, this sets them all to ones
        act = np.array([1]*len(SOM[0,0]))
    if output == 0:  # if we want "hard" output
        g, h, eucl_dist = find_BMU(SOM, vector, act)
        return SOM[g, h]
    elif output == 1:  # if we want "soft" output
        activity_n = activity_vec(vector, SOM, act, act_norm=1, sens=sens)
        sm = np.zeros(len(SOM[0,0]))
        for i, row in enumerate(activity_n):
            for j, neuron in enumerate(row):
                sm += np.multiply(neuron, SOM[i, j])
        return sm

def test(tonic_no, SOM, act=None, output=0, sens=1, decay=0.9):
    if np.all(act == None):  # if the attention weights are not specified, this sets them
        act = np.array([1] * len(SOM[0, 0]))  # to ones for the tonic, context and recent part
        act[300:400] = np.array([0]*100)  # and to zeros for the next part
    steps = 0
    max_steps = 15
    tonic = np.zeros(100)
    ctx = np.zeros(100)
    rec = np.zeros(100)
    tonic[tonic_no] = 1
    letter = None
    letters = []
    while (letter == None or letter != ".") and steps < max_steps:
        steps += 1
        nxt = np.zeros(100)
        pred = act_output(np.concatenate((tonic, ctx, rec, nxt), axis=0), SOM, act=act, output=output, sens=sens)
        i = np.argmax(pred[300:400])
        nxt[i] = 1
        if i == 99:
            letter = "."
        else:
            letter = chr(i+97)  # which letter it was predicted
            letters.append(letter)
        ctx = decay * ctx + rec  # we add recent to the context
        rec = nxt  # we add next to the recent
    return "".join(letters)  # the list is joint in a string

# an example of how to test the data
for i in range(13):
    t = test(i, SOM, output=1, sens=2)
    print(t)
