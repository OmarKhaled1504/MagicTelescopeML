import random


class Instance:
    def __init__(self, fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist,
                 gorh):  # class is reserved word so we're using gorh (gamma or hadron)
        self.flength = fLength
        self.fwidth = fWidth
        self.fSize = fSize
        self.fConc = fConc
        self.fConc1 = fConc1
        self.fAsym = fAsym
        self.fM3Long = fM3Long
        self.fM3Trans = fM3Trans
        self.fAlpha = fAlpha
        self.fDist = fDist
        self.gorh = gorh


if __name__ == "__main__":
    instances = []
    training_set = []
    test_set = []

    file = open("magic04.data", "r")
    for line in file:
        fields = line.split(",")
        instances.append(
            Instance(float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]),
                     float(fields[5]), float(fields[6]), float(fields[7]), float(fields[8]),
                     float(fields[9]), fields[10].replace("\n", '')))
    file.close()
    i = 5644                                                                                # 5644 is the number of classes (g) that should be randomly removed for the two classes to be balanced
    while i > 0:
        j = random.randint(0, 12331)
        if instances[j].gorh == 'g':
            del instances[j]
            i -= 1
    i = 9363                                                                                # 70% of the data set for training
    while i > 0:

        j = (random.randint(0, 13375)) % (len(instances))
        training_set.append(instances.pop(j))
        i -= 1
    i = 4013                                                                                # 30% for testing
    while i > 0:
        j = (random.randint(0, 4012)) % (len(instances))
        test_set.append(instances.pop(j))
        i -= 1
