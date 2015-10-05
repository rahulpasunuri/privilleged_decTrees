train = open("original/heart_train.csv", "r").readlines()
test = open("original/heart_test.csv", "r").readlines()

priv_train = open("original/heart_priv_train.csv", "w")
pruned_train = open("original/heart_pruned_train.csv", "w")

priv_test = open("original/heart_priv_test.csv", "w")
pruned_test = open("original/heart_pruned_test.csv", "w")

def split(orig, pruned, priv):
    for l in orig:
        l=l.strip()
        words = l.split(",")
        privColumns = [11, 12]
        privInfo = []
        pruneInfo = []
        for i in range(len(words)):
            if i in privColumns:
                privInfo.append(words[i])
            else:
                pruneInfo.append(words[i])
        privInfo.append(words[len(words) -1])
        priv.write(",".join(privInfo)+"\n")
        pruneInfo.append(words[i])
        pruned.write(",".join(pruneInfo)+"\n")
        
split(train, pruned_train, priv_train,)
split(test, pruned_test, priv_test)
