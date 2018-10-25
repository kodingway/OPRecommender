import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt

def sortDist(tup):
    return tup[1]

def sortValue(tup):
    return -evalFunc(tup[0],testUsers)

def sortRatio(tup):
    if(tup[1]==0):
        return math.inf
    else:
        return -evalFunc(tup[0],testUsers)/tup[1]

def readGraph(city):
    with open("Dataset/" + city + "/Distances.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        global graph
        graph = []
        tempList = []
        currPOI = 1
        for row in csvReader:
            if (int(row[0]) != currPOI):
                graph.append(tempList)
                tempList = []
                currPOI += 1
            tempList.append((int(row[1]) - 1, float(row[2])))
        graph.append(tempList)

        global graphAdj
        graphAdj = np.zeros([len(graph), len(graph)])
        csvFile.seek(0)
        for row in csvReader:
            graphAdj[int(row[0]) - 1, int(row[1]) - 1] = float(row[2])


def readUsers(city):
    with open("Dataset/" + city + "/Users_Profiles.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        userId = 0
        userFeatures = 0
        for row in csvReader:
            userId += 1
            userFeatures = len(row)

        csvFile.seek(0)
        global users
        users = np.zeros([userId, userFeatures - 1])
        userId = 0
        for row in csvReader:
            for i in range(8):
                users[userId, i] = float(row[i])
            userId += 1


def readPOIs(city):
    with open("Dataset/" + city + "/POIs_Vectors.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        global pois
        pois = np.zeros([len(graph), len(users[0])])
        for row in csvReader:
            for i in range(8):
                pois[int(row[0]) - 1, i] = float(row[i + 1])


def readStayTimes(city):
    with open("Dataset/" + city + "/POIs_Vectors.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        global stayTime
        stayTime = np.zeros(len(graph))
        for row in csvReader:
            stayTime[int(row[0]) - 1] = float(row[1])


city = "Rome"
readGraph(city)
readUsers(city)
readPOIs(city)
readStayTimes(city)


def satisfactionSum(poiId,testUsers):
    return sum(testUsers.dot(pois[poiId]))


def satisfactionMin(poiId,testUsers):
    return min(testUsers.dot(pois[poiId]))


def satisfactionFair(poiId,testUsers):
    tempSat = testUsers.dot(pois[poiId])
    return np.mean(tempSat) - 0.5 * np.std(tempSat)

B = 5  # Budget (distance)
while True:
    (s, t) = random.sample(range(len(graph)), 2)
    if graphAdj[s, t] <= B:
        break


def bestPath(s, t, sortFunc, testUsers):

    path = [s]
    totalProfit = evalFunc(s,testUsers)
    totalDistance = 0
    lastVisit = s
    oldPath = []
    while (totalDistance + graphAdj[lastVisit, t] < B):
        oldPath = path.copy()
        graph[lastVisit].sort(key=sortFunc)
        for neigh in graph[lastVisit]:
            currProfit = evalFunc(neigh[0],testUsers)

            if ((currProfit > 0) and (neigh[0] not in path) and (
                    totalDistance + neigh[1] + graphAdj[neigh[0], t] <= B)):
                path.append(neigh[0])
                totalProfit += currProfit
                lastVisit = neigh[0]
                totalDistance += neigh[1]
                break
        if (path == oldPath):
            break

    return (path, totalProfit, totalDistance)

evalFunc=satisfactionSum

bestDistance=[]
for rep in range(500):

    score=[]
    for k in range(1,21):
        testUsers = random.sample(list(users),k)
        testUsers = np.array([np.array(xi) for xi in testUsers])
        score.append(bestPath(s, t, sortDist,testUsers)[1])
    if rep==0:
        bestDistance=score
    else:
        bestDistance=np.add(bestDistance,score)

bestDistance[:] = [x/500 for x in bestDistance]

bestValue=[]
print("Hey")
for rep in range(500):
    score=[]
    for k in range(1,21):
        testUsers = random.sample(list(users),k)
        testUsers = np.array([np.array(xi) for xi in testUsers])
        score.append(bestPath(s, t, sortValue, testUsers)[1])
    if rep==0:
        bestValue=score
    else:
        bestValue=np.add(bestValue,score)

bestValue[:] = [x/500 for x in bestValue]

bestRatio=[]
print("Hey there")
for rep in range(500):
    score=[]
    for k in range(1,21):
        testUsers = random.sample(list(users),k)
        testUsers = np.array([np.array(xi) for xi in testUsers])
        score.append(bestPath(s, t, sortRatio, testUsers)[1])
    if rep == 0:
        bestRatio=score
    else:
        bestRatio=np.add(bestRatio,score)

bestRatio[:] = [x/500 for x in bestRatio]

plt.plot(bestDistance,'r^', bestValue, 'gs', bestRatio, 'b--')
plt.savefig('Distance+Value+Ratio')
