from collections import defaultdict
import torch

def loadData(File, splitMark, based):
    DataSet = defaultdict(list) 
    max_u_id = -1
    max_i_id = -1
    
    for line in open(File):
        userId, itemId, rating, _ = line.strip().split(splitMark)
        userId = int(userId) - 1 
        itemId = int(itemId) - 1 
        DataSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    for u, i_list in DataSet.items():
        i_list.sort()

    userCount = max_u_id + 1
    itemCount = max_i_id + 1

    if based == 'Train':
        print('Training data loading done: %d users, %d items' % (userCount, itemCount))
    else:
        print('Test data loading done: %d users, %d items' % (userCount, itemCount))
    return DataSet, userCount, itemCount  

def to_Tensor(trainSet, testSet, userCount, itemCount):
    # assume that the default is itemBased

    testMaskDict = defaultdict(lambda: [0] * itemCount) 
    trainDict = defaultdict(lambda: [0] * itemCount)

    for userID, item_list in trainSet.items():
        for itemID in item_list:
            testMaskDict[userID][itemID] = -99999
            trainDict[userID][itemID] = 1.0

    trainVector = []
    for userID in range(userCount):
        trainVector.append(trainDict[userID])

    userList_test = list(testSet.keys())    
    testMaskVector = []
    for userID in userList_test:
        testMaskVector.append(testMaskDict[userID])

    print("Converting to vectors done...")

    return torch.Tensor(trainVector), torch.Tensor(testMaskVector)

def computeTopNAccuracy(groundTruth, result, topN):
    result = result.tolist()
    for i in range(len(result)):
        result[i] = (result[i],i)
    result.sort(key=lambda x:x[0], reverse=True)
    top5 = 5
    top10 = 10
    hit5, hit10, hit20 = 0, 0, 0

    for i in range(top5):
        if(result[i][1] in groundTruth):
            hit5 = hit5 + 1
    for i in range(top10):
        if(result[i][1] in groundTruth):
            hit10 = hit10 + 1
    for i in range(topN):
        if(result[i][1] in groundTruth):
            hit20 = hit20 + 1
    return hit5/len(groundTruth), hit5/top5, hit10/len(groundTruth), hit10/top10, hit20/len(groundTruth), hit20/topN