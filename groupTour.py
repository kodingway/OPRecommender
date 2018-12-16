import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt

def readGraph(city):
    with open("Dataset/" + city + "/Distances.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        walkSpeed = 10/60  # (km/min)

        global graph
        graph = {}
        currPOI = 0
        for row in csvReader:
            if (currPOI!=int(row[0])):
                graph[int(row[0])]=[]
                currPOI=int(row[0])
            graph[int(row[0])].append((int(row[1]), float(row[2])/walkSpeed))

        global graphAdj
        graphAdj = np.zeros([len(graph)+1, len(graph)+1])
        csvFile.seek(0)
        for row in csvReader:
            graphAdj[int(row[0]), int(row[1])] = float(row[2])/walkSpeed

def readUsers(city):
    with open("Dataset/" + city + "/Users_Profiles.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        global users
        users = {}
        userId = 1
        for row in csvReader:
            users[userId]=list(map(float,row[:8]))
            userId += 1

def readPOIs(city):
    with open("Dataset/" + city + "/POIs_Vectors.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        global pois
        pois = {}
        for row in csvReader:
            pois[int(row[0])]=list(map(float,row[1:9]))

def readStayTimes(city):
    with open("Dataset/" + city + "/POIs_StayTime.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        global stayTime
        stayTime={}
        for row in csvReader:
            stayTime[int(row[0])] = float(row[1])/60 # (min)

city = "Rome"
readGraph(city)
readUsers(city)
readPOIs(city)
readStayTimes(city)

stayMax=max(stayTime, key=stayTime.get)
stayMin=min(stayTime, key=stayTime.get)

for key in range(1,len(graph)+1):
    if key not in list(pois.keys()):
        pois[key]=pois[1]  #some pois don't have vectors
    if key not in list(stayTime.keys()):
        stayTime[key]=random.uniform(stayTime[stayMin],stayTime[stayMax]) #a lot of pois don't have stay times

def satisfactionSum(poiId, testUsers):
    return sum(np.matmul(testUsers, pois[poiId]))

def satisfactionMin(poiId, testUsers):
    return min(np.matmul(testUsers, pois[poiId]))

def satisfactionFair(poiId, testUsers):
    tempSat=np.matmul(testUsers, pois[poiId])
    return np.mean(tempSat) - 0.5 * np.std(tempSat)

def pathCost(path):
    result=0
    for i,node in enumerate(path[0:-1]):
        result+=stayTime[node]+graphAdj[node,path[i+1]]
    return(result)

def pathProfit(path,testUsers):
    result=0
    for node in path[1:-1]:
        result+=evalFunc(node,testUsers)
    return(result)

########## BEST VALUE HEURISTIC ##########
##########################################
##########################################
def bestValuePath(s, t, testUsers):

    path=[s,t]
    while True:
        bestPath = path.copy()
        bestValue = 0
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                potPath = path.copy()
                potPath.insert(-1,neigh)
                if (pathProfit(potPath,testUsers)>bestValue) and (pathCost(potPath) <= B):
                    bestPath = potPath.copy()
                    bestValue = pathProfit(bestPath,testUsers)
        if bestPath == path:
            return path
        path = bestPath.copy()

########## BEST DISTANCE HEURISTIC ##########
#############################################
#############################################
def bestDistancePath(s, t):

    path=[s,t]
    while True:
        closestDistance=float("inf")
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                if (graphAdj[path[-2],neigh]+stayTime[neigh]<closestDistance):
                    potPath=path.copy()
                    potPath.insert(-1,neigh)
                    if (pathCost(potPath)<=B):
                        closestDistance=graphAdj[path[-2],neigh]+stayTime[neigh]
                        bestPath = potPath.copy()
        if bestPath == path:
            return path
        path = bestPath.copy()

########## BEST RATIO HEURISTIC ##########
##########################################
##########################################
def bestRatioPath(s, t, testUsers):

    path=[s,t]
    while True:
        bestRatio=0
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                potPath = path.copy()
                potPath.insert(-1,neigh)
                if (pathProfit(potPath,testUsers)/pathCost(potPath)>bestRatio) and (pathCost(potPath) <= B):
                    bestPath = potPath.copy()
                    bestRatio = pathProfit(potPath,testUsers)/pathCost(potPath)
        if bestPath == path:
            return path
        path = bestPath.copy()


########## BEST RATIO+ HEURISTIC ##########
###########################################
###########################################
def bestRatioPlusPath(s, t, testUsers):

    path=[s,t]
    while True:
        bestRatio=0
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                potPath = path.copy()
                potPath.insert(-1,neigh)
                if (pathProfit(potPath,testUsers)/pathCost(potPath)>bestRatio) and (pathCost(potPath) <= B):
                    bestPath = potPath.copy()
                    bestRatio = pathProfit(potPath,testUsers)/pathCost(potPath)
        if bestPath == path:
            break
        path = bestPath.copy()

    while True:
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                for nodeI in range(1,len(path)-1):
                    potPath = path.copy()
                    potPath[nodeI] = neigh
                    if (pathProfit(potPath,testUsers)/pathCost(potPath)>pathProfit(bestPath,testUsers)/pathCost(bestPath)) and (pathCost(potPath) <= B):
                        bestPath = potPath.copy()
        if bestPath == path:
            return path
        path = bestPath.copy()


########## BEST RATIO++ HEURISTIC ##########
############################################
############################################
def bestRatioPlusPlusPath(s, t, testUsers):

    path=[s,t]
    while True:
        bestRatio=0
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                potPath = path.copy()
                potPath.insert(-1,neigh)
                if (pathProfit(potPath,testUsers)/pathCost(potPath)>bestRatio) and (pathCost(potPath) <= B):
                    bestPath = potPath.copy()
                    bestRatio = pathProfit(potPath,testUsers)/pathCost(potPath)
        if bestPath == path:
            break
        path = bestPath.copy()

    while True:
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            if neigh not in path:
                for nodeI in range(1,len(path)-1):
                    potPath = path.copy()
                    potPath[nodeI] = neigh
                    if (pathProfit(potPath,testUsers)/pathCost(potPath)>pathProfit(bestPath,testUsers)/pathCost(bestPath)) and (pathCost(potPath) <= B):
                        bestPath = potPath.copy()
        if bestPath == path:
            break
        path = bestPath.copy()

    while True:
        bestPath = path.copy()
        for neigh in range(1,len(graphAdj)):
            for neigh2 in range(1,len(graphAdj)):
                if (neigh!=neigh2) and (neigh not in path) and (neigh2 not in path):
                    for nodeI in range(1,len(path)-1):
                        potPath = path.copy()
                        potPath[nodeI] = neigh
                        potPath.insert(-1,neigh2)
                        if (pathProfit(potPath,testUsers)/pathCost(potPath)>pathProfit(bestPath,testUsers)/pathCost(bestPath)) and (pathCost(potPath) <= B):
                            bestPath = potPath.copy()
        if bestPath == path:
            return path
        path = bestPath.copy()

########## HEURISTICS SIMULATION & COMPARISON ##########
########################################################
########################################################
evalFunc = satisfactionSum
totalReps = 10
B = 420  # Budget (minutes)
klist=[1,2,5,10,20]

# bestDistance = []
# bestValue=[]
# bestRatio=[]
# bestRatioPlus=[]
# #bestRatioPlusPlus=[]

# for rep in range(totalReps):

    # while True:
    #     (s, t) = random.sample(range(1,len(graph)+1), 2)
    #     if pathCost([s,t]) <= B:
    #         break

#     print(rep)
#     distanceScore = []
#     valueScore = []
#     ratioScore = []
#     ratioPScore = []
#     #ratioPPScore = []
#     distPath = bestDistancePath(s,t)
#     for k in klist:
#         testUsers = []
#         testSet = random.sample(range(1,len(users)+1), k)
#         for l in testSet:
#             testUsers.append(users[l])
#         testUsers = np.array([np.array(xi) for xi in testUsers])

#         distanceScore.append(pathProfit(distPath,testUsers))
#         valPath = bestValuePath(s, t, testUsers)
#         ratPath = bestRatioPath(s, t, testUsers)
#         ratPPath = bestRatioPlusPath(s, t, testUsers)
        
#         valueScore.append(pathProfit(valPath, testUsers))
#         ratioScore.append(pathProfit(ratPath, testUsers))
#         ratioPScore.append(pathProfit(ratPPath, testUsers))
#         #ratioPPScore.append(bestRatioPlusPlusPath(s, t, testUsers)[1])

#     if rep == 0:
#         bestDistance = distanceScore
#         bestValue = valueScore
#         bestRatio = ratioScore
#         bestRatioPlus = ratioPScore
#         #bestRatioPlusPlus = ratioPPScore
#     else:
#         bestDistance = np.add(bestDistance, distanceScore)
#         bestValue = np.add(bestValue, valueScore)
#         bestRatio = np.add(bestRatio, ratioScore)
#         bestRatioPlus = np.add(bestRatioPlus, ratioPScore)
#         #bestRatioPlusPlus = np.add(bestRatioPlusPlus, ratioPPScore)

# bestDistance[:] = [x/totalReps for x in bestDistance]
# bestValue[:] = [x/totalReps for x in bestValue]
# bestRatio[:] = [x/totalReps for x in bestRatio]
# bestRatioPlus[:] = [x/totalReps for x in bestRatioPlus]
# #bestRatioPlusPlus[:] = [x/totalReps for x in bestRatioPlusPlus]

# plt.figure()
# plt.plot(klist,bestDistance, marker='v', color='purple', linestyle='--')
# plt.plot(klist,bestValue, marker='^', color='aqua', linestyle='--')
# plt.plot(klist,bestRatio, marker='D', color='r', linestyle='--')
# plt.plot(klist,bestRatioPlus, marker='s', color='fuchsia', linestyle='--')
# plt.legend(['bestDistance', 'bestValue','bestRatio', 'bestRatio+'])
# plt.xticks(range(1,21))
# plt.xlabel('Group size')
# plt.ylabel('Average Solution Value')
# plt.title('Satisfaction Sum')
# plt.tight_layout()
# plt.show()

########## K-MEANS CLUSTERING ##########
########################################
########################################
def kmeans(k, testSet):
    
    initSet = random.sample(testSet, k) #Forgy initialization
    centroids = [users[i] for i in initSet] 
    clusterIds=[]

    for i in range(k):
        clusterIds.append([])
    while True:
        oldClusterIds = clusterIds.copy()
        clusterIds=[]
        for i in range(k):
            clusterIds.append([])
        for userId in testSet:
            distances=[]
            for centr in centroids:
                distances.append(np.linalg.norm(np.subtract(centr,users[userId])))
            clusterIds[distances.index(min(distances))].append(userId)
        for i in range(k):
            centroids[i]=list(np.mean([users[id] for id in clusterIds[i]],axis=0))
        if oldClusterIds==clusterIds:
            break
    return clusterIds


m=200 #Number of users
totalScores = []
for kaka in range(100):
    print(kaka)
    while True:
        (s, t) = random.sample(range(1,len(graph)+1), 2)
        if pathCost([s,t]) <= B:
            break
    testSet = random.sample(range(1,len(users)+1), m)
    kscores=[]
    for k in [1,2,5,10,20]:
        print(kaka,k)
        scores = []
        for rep in range(10):
            clusterIds = kmeans(k,testSet)
            totalProfit=0
            for cluster in clusterIds:
                testUsers=[]
                for l in cluster:
                    testUsers.append(users[l])
                testUsers = np.array([np.array(xi) for xi in testUsers])
                clusterPath = bestRatioPath(s, t, testUsers)
                totalProfit += pathProfit(clusterPath, testUsers)
            scores.append(totalProfit)
        kscores.append(max(scores))
    totalScores.append(kscores)    