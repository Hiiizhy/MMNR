import numpy as np


def generateBatchSamples(dataLoader, batchIdx, config, isEval):
    samples, sampleLen, userhis, itemhis, target = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)

    maxLenSeq = max([len(userLen) for userLen in sampleLen])  # max length of sequence
    maxLenBas = max([max(userLen) for userLen in sampleLen])  # max length of basket

    # pad users
    paddedSamples = []
    paddedDecays = []
    lenList = []
    for user in samples:
        trainU = user[:-1]

        paddedU = []
        decayU = []
        lenList.append([len(trainU) - 1])
        decayNum = len(trainU) - 1
        for eachBas in trainU:
            paddedBas = eachBas + [config.padIdx] * (maxLenBas - len(eachBas))
            paddedU.append(paddedBas)  # [batch, maxLenBas]
            decayU.append(config.decay ** decayNum)
            decayNum -= 1
        paddedU = paddedU + [[config.padIdx] * maxLenBas] * (maxLenSeq - len(paddedU))
        decayU = decayU + [0] * (maxLenSeq - len(decayU))
        paddedSamples.append(paddedU)  # [batch, maxLenSeq]
        paddedDecays.append(decayU)  # [batch, maxLenSeq]

    return np.asarray(paddedSamples), np.asarray(paddedDecays).reshape(len(samples), -1, 1), userhis, itemhis, target
