import torch
import numpy as np
import math
from utils import generateBatchSamples


def evaluate_ranking(model, dataLoader, config, device, isTrain):
    evalBatchSize = config.batch_size

    if isTrain:
        numUser = dataLoader.numValid
    else:
        numUser = dataLoader.numTest

    if numUser % config.batch_size == 0:
        numBatch = numUser // evalBatchSize
    else:
        numBatch = numUser // evalBatchSize + 1

    idxList = [i for i in range(numUser)]

    Recall = []
    NDCG = []

    for batch in range(numBatch):
        start = batch * evalBatchSize
        end = min(batch * evalBatchSize + evalBatchSize, numUser)

        batchList = idxList[start:end]

        samples, decays, userhis, itemhis, target = generateBatchSamples(dataLoader, batchList, config, isEval=1)

        samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
        decays = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
        userhis = torch.from_numpy(userhis).type(torch.FloatTensor).to(device)
        itemhis = torch.from_numpy(itemhis).type(torch.FloatTensor).to(device)

        with torch.no_grad():
            scores, _ = model.forward(samples, decays, userhis, itemhis, device)

        # get the index of top 40 items
        predIdx = torch.topk(scores, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList = target
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            targetList += target

    for k in [5, 10]:
        recall = calRecall(targetList, predIdxArray, k)
        Recall.append(recall)
        NDCG.append(calNDCG(targetList, predIdxArray, k))
    return Recall, NDCG


def calRecall(target, pred, k):
    assert len(target) == len(pred)
    sumRecall = 0
    for i in range(len(target)):
        gt = set(target[i])
        ptar = set(pred[i][:k])

        if len(gt) == 0:
            print('Error')

        sumRecall += len(gt & ptar) / float(len(gt))

    recall = sumRecall / float(len(target))

    return recall


def calNDCG(target, pred, k):
    assert len(target) == len(pred)
    sumNDCG = 0
    for i in range(len(target)):
        valK = min(k, len(target[i]))
        gt = set(target[i])
        idcg = calIDCG(valK)
        dcg = sum([int(pred[i][j] in gt) / math.log(j + 2, 2) for j in range(k)])
        sumNDCG += dcg / idcg

    return sumNDCG / float(len(target))


# the gain is 1 for every hit, and 0 otherwise
def calIDCG(k):
    return sum([1.0 / math.log(i + 2, 2) for i in range(k)])
