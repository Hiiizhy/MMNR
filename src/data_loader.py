from scipy import sparse
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import scipy
from sklearn.preprocessing import normalize
import time


class dataLoader():
    def __init__(self, config):
        dataRoot = '../data/' + config.dataset + '.pkl'
        with open(dataRoot, 'rb') as f:
            dataDict = pickle.load(f)

        user2item = self.generate_user_list(dataDict)

        numRe, user2idx = self.generate_sets(user2item)
        self.numItems = self.get_num_items() + 1

        print("num_users_removed   %d" % numRe)
        print("num_valid_users   %d" % len(self.testList))
        print("num_items   %d" % self.numItems)

        # to generate his
        self.numTrain, self.numValid, self.numTrainVal, self.numTest = len(self.testList), len(self.testList), len(
            self.testList), len(self.testList)
        # same number of items for training and testing
        self.numItemsTrain, self.numItemsTest = self.numItems, self.numItems
        # same id in training, validation and testing
        self.valid2train = {}
        self.test2trainVal = {}
        for i in range(len(self.trainList)):
            self.valid2train[i] = i
            self.test2trainVal[i] = i

        if config.isTrain:
            self.lenTrain = self.generateLens(self.trainList)
            self.lenVal = self.generateLens(self.validList)
        else:  # Test
            self.lenTrainVal = self.generateLens(self.trainValList)
            self.lenTest = self.generateLens(self.testList)

        if config.isTrain:
            self.userhisMatTra, self.itemhisMatTra, self.hisMatTra, self.tarMatTra = self.generateHis(self.trainList,
                                                                                                      isTrain=1,
                                                                                                      isEval=0)
            self.userhisMatVal, self.itemhisMatVal, self.hisMatVal, self.tarMatVal = self.generateHis(self.validList,
                                                                                                      isTrain=1,
                                                                                                      isEval=1)

        else:
            self.userhisMatTraVal, self.itemhisMatTraVal, self.hisMatTraVal, self.tarMatTraVal = self.generateHis(
                self.trainValList, isTrain=0, isEval=0)
            self.userhisMatTest, self.itemhisMatTest, self.hisMatTest, self.tarMatTest = self.generateHis(self.testList,
                                                                                                          isTrain=0,
                                                                                                          isEval=1)

    # def generate_user_list(self, trainDict, validDict, testDict):
    def generate_user_list(self, dataDict):
        all_users = list(dataDict.keys())
        user2item = {}
        for user in all_users:
            user2item[user] = dataDict.get(user, [])
        return user2item

    def generate_sets(self, user2item):
        self.trainList = []
        self.validList = []
        self.trainValList = []
        self.testList = []
        count = 0
        count_remove = 0
        user2idx = {}

        for user in user2item:
            if len(user2item[user]) < 4:  # train>=2, valid=1, test=1
                count_remove += 1
                continue
            user2idx[user] = count
            count += 1
            self.trainList.append(user2item[user][:-2])
            self.validList.append(user2item[user][:-1])
            self.trainValList.append(user2item[user][:-1])
            self.testList.append(user2item[user])
        return count_remove, user2idx

    def get_num_items(self):
        numItem = 0
        for baskets in self.testList:
            # all the baskets of users
            for basket in baskets:
                for item in basket:
                    numItem = max(item, numItem)

        return numItem

    def batchLoader(self, batchIdx, isTrain, isEval):
        if isTrain and not isEval:
            train = [self.trainList[idx] for idx in batchIdx]
            trainLen = [self.lenTrain[idx] for idx in batchIdx]
            userhis = self.userhisMatTra[batchIdx, :].todense()
            itemhis = self.itemhisMatTra[batchIdx, :].todense()
            target = self.tarMatTra[batchIdx, :].todense()
        elif isTrain and isEval:
            train = [self.validList[idx] for idx in batchIdx]
            trainLen = [self.lenVal[idx] for idx in batchIdx]
            userhis = self.userhisMatVal[batchIdx, :].todense()
            itemhis = self.itemhisMatVal[batchIdx, :].todense()
            target = [self.tarMatVal[idx] for idx in batchIdx]
        elif not isTrain and not isEval:
            train = [self.trainValList[idx] for idx in batchIdx]
            trainLen = [self.lenTrainVal[idx] for idx in batchIdx]
            userhis = self.userhisMatTraVal[batchIdx, :].todense()
            itemhis = self.itemhisMatTraVal[batchIdx, :].todense()
            target = self.tarMatTraVal[batchIdx, :].todense()
        else:
            train = [self.testList[idx] for idx in batchIdx]
            trainLen = [self.lenTest[idx] for idx in batchIdx]
            userhis = self.userhisMatTest[batchIdx, :].todense()
            itemhis = self.itemhisMatTest[batchIdx, :].todense()
            target = [self.tarMatTest[idx] for idx in batchIdx]

        return train, trainLen, userhis, itemhis, target

    def generateLens(self, userList):
        # list of list of lens of baskets
        lens = []
        # pre-calculate the len of each sequence and basket
        for user in userList:
            lenUser = []
            # the last bas is the traget to calculate errors
            trainEUser = user[:-1]
            for bas in trainEUser:
                lenUser.append(len(bas))
            lens.append(lenUser)

        return lens

    def generateHis(self, userList, isTrain, isEval):

        # pre generate the history vector and target vector for each user
        if isTrain and not isEval:
            hisMat = np.zeros((self.numTrain, self.numItemsTrain))
        elif isTrain and isEval:
            hisMat = np.zeros((self.numValid, self.numItemsTrain))
        elif not isTrain and not isEval:
            hisMat = np.zeros((self.numTrainVal, self.numItemsTest))
        else:
            hisMat = np.zeros((self.numTest, self.numItemsTest))

        if not isEval and isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTrain))
        elif not isEval and not isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTest))
        else:
            tarMat = []

        for i in range(len(userList)):
            trainUser = userList[i][:-1]
            targetUser = userList[i][-1]

            for Bas in trainUser:
                for item in Bas:
                    hisMat[i, item] += 1
            if not isEval:
                for item in targetUser:
                    tarMat[i, item] = 1
            else:
                tarMat.append(targetUser)

        # convert numpy to csr matrix
        hisMat = csr_matrix(hisMat)
        if not isEval:
            tarMat = csr_matrix(tarMat)

        userHisMat = normalize(hisMat, norm='l1', axis=1, copy=False)
        itemHisMat = normalize(hisMat, norm='l1', axis=0, copy=False)

        return userHisMat, itemHisMat, hisMat, tarMat
