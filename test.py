# -*- coding = utf-8 -*-
# @Time : 2021/4/21 15:22
# @Author : HEYME
# @File : test.py
# @Software : PyCharm
from sklearn.ensemble import RandomForestClassifier
import math


domainlist = []
class Domain:
    def __init__(self, _name, _label, _len, _digit, _entropy):
        self.name = _name
        self.label = _label
        self.length = _len
        self.digitNum = _digit
        self.entropy = _entropy

    def returnData(self):
        return [self.length, self.digitNum, self.entropy]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1


def calDigitNum(domain):
    count = 0
    for i in domain:
        if i.isdigit():
            count += 1
    return count


def calEntropy(domain):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = domain.lower()
    for i in range(len(domain)):
        if domain[i].isalpha():
            letter[ord(domain[i]) - ord('a')] += 1
            sum += 1
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return round(h, 3)


def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(name)
            digitNum = calDigitNum(name)
            entropy = calEntropy(name)
            domainlist.append(Domain(name, label, length, digitNum, entropy))


def predictData(filename, Classifier):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            length = len(name)
            digitNum = calDigitNum(name)
            entropy = calEntropy(name)
            res = Classifier.predict([[length, digitNum, entropy]])
            if int(res) == 0:
                print ("%s,notdga" % name)
            else:
                print ("%s,dga" % name)


def main():
    print("Initialize Raw Objects...")
    initData("train.txt")
    featureMatrix = []
    labelList = []
    print("Initialize Matrix...")
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    print(featureMatrix)
    print("Begin Training...")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    print("Begin Predicting:")
    predictData("test.txt", clf)



if __name__ == '__main__':
    main()

