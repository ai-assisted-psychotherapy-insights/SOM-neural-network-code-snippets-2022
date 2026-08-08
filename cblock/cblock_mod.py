import numpy as np

# import the data from the text file
names = np.genfromtxt("names.txt", delimiter=", ", dtype="U")

def data(names, decay1=0.9, decay2=0.7, m=1):
    for i, name in enumerate(names):
        tonic = np.zeros(100)
        ctx = np.zeros(100)
        rec = np.zeros(100)
        nxt = np.zeros(100)
        for j, let in enumerate(name):
            ctx = decay1*ctx + rec  # we add the recent to context (and multiply the previous context with a decay constant)
            rec = nxt  # we add the previous next to recent
            nxt = np.zeros(100)  # we reset the next value to zeros
            nxt[ord(let.lower())-97] = 1  # we establish a new next with the current letter
            tonic += m*nxt
            m = decay2*m
            conc = np.concatenate((tonic, ctx, rec, nxt), axis=0)  # concatenate all these vectors
            if i == j == 0:  # if this is the first person and the first letter of the person
                dat = np.append(let.lower(), conc)  # save the letter and the concatenation in a new array
            else:
                new_d = np.append(let.lower(), conc)  # add also the information of the letter
                dat = np.vstack((dat, new_d))  # add the concatenation to the previous
        ctx = decay1*ctx + rec  # for the last letter
        rec = nxt
        nxt = np.append([0]*99, 1)  # this is the stopping condition (the last value is 1)
        tonic += m*nxt
        conc = np.concatenate((tonic, ctx, rec, nxt), axis=0)
        new_d = np.append(".", conc)
        dat = np.vstack((dat, new_d))  # save also the last information about the stopping condition
        m = 1
    return dat

# an example how to get the training data and save it
dat = data(names)
# save all the data
np.savetxt("train_data_mod.txt", dat, delimiter=",", fmt="%s")  # the fmt argument is for specifying it is a string
