import numpy as np
import csv
import random
import math

def readGraph(city):
    with open("Dataset/" + city + "/Distances.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        walkSpeed = 4.5/60  # (km/min)

        global graph
        graph = {}
        tempList = []
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
        stayTime[key]=0.0#random.uniform(stayTime[stayMin],stayTime[stayMax]) #a lot of pois don't have stay times

def satisfactionSum(poiId, testUsers):
    return sum(np.matmul(testUsers, pois[poiId]))

def satisfactionMin(poiId, testUsers):
    return min(np.matmul(testUsers, pois[poiId]))

def satisfactionFair(poiId, testUsers):
    tempSat=np.matmul(testUsers, pois[poiId])
    return np.mean(tempSat) - 0.5 * np.std(tempSat)

def bestValuePath(s, t, testUsers):

    def sortValue(edge):
        return evalFunc(edge[0], testUsers)

    path=[s]
    totalProfit = evalFunc(s, testUsers)
    totalDistance = 0
    lastVisit = s
    oldPath = []
    while True:
        oldPath = path.copy()
        graph[lastVisit].sort(key=sortValue,reverse=True)
        for neigh in graph[lastVisit]:
            currProfit = evalFunc(neigh[0], testUsers)
            if ((currProfit > 0) and (neigh[0]!=t) and (neigh[0] not in path) and (
                    totalDistance + neigh[1] + stayTime[neigh[0]] + graphAdj[neigh[0], t] <= B)):
                path.append(neigh[0])
                totalProfit += currProfit
                lastVisit = neigh[0]
                totalDistance += neigh[1] + stayTime[neigh[0]]
                break
        if path == oldPath:
            break

    path.append(t)
    totalProfit += evalFunc(t, testUsers)
    totalDistance += graphAdj[lastVisit, t] + stayTime[t]
    return (path, totalProfit, totalDistance)

def bestDistancePath(s, t, testUsers):

    def sortDist(edge):
        return edge[1]

    path=[s]
    totalProfit = evalFunc(s, testUsers)
    totalDistance = 0
    lastVisit = s
    oldPath = []
    while True:
        oldPath = path.copy()
        graph[lastVisit].sort(key=sortDist)
        for neigh in graph[lastVisit]:
            currProfit = evalFunc(neigh[0], testUsers)
            if ((currProfit > 0) and (neigh[0]!=t) and (neigh[0] not in path) and (
                    totalDistance + neigh[1] + stayTime[neigh[0]] + graphAdj[neigh[0], t] <= B)):
                path.append(neigh[0])
                totalProfit += currProfit
                lastVisit = neigh[0]
                totalDistance += neigh[1] + stayTime[neigh[0]]
                break
        if path == oldPath:
            break

    path.append(t)
    totalProfit += evalFunc(t, testUsers)
    totalDistance += graphAdj[lastVisit, t] + stayTime[t]
    return (path, totalProfit, totalDistance)

def bestRatioPath(s, t, testUsers):
    
    def sortRatio(edge):
        potDistance = totalDistance + edge[1] + stayTime[edge[0]]
        potProfit = totalProfit + evalFunc(edge[0], testUsers)
        if potDistance == 0:
            return float("inf")
        else:
            return potProfit/potDistance
    
    path=[s]
    totalProfit = evalFunc(s, testUsers)
    totalDistance = 0
    lastVisit = s
    oldPath = []
    while True:
        oldPath = path.copy()
        graph[lastVisit].sort(key=sortRatio,reverse=True)
        for neigh in graph[lastVisit]:
            currProfit = evalFunc(neigh[0], testUsers)
            if ((currProfit > 0) and (neigh[0]!=t) and (neigh[0] not in path) and (
                    totalDistance + neigh[1] + stayTime[neigh[0]] + graphAdj[neigh[0], t] <= B)):
                path.append(neigh[0])
                totalProfit += currProfit
                lastVisit = neigh[0]
                totalDistance += neigh[1] + stayTime[neigh[0]]
                break
        
        #print(totalProfit/totalDistance,totalProfit)
        if path == oldPath:
            break

    path.append(t)
    totalProfit += evalFunc(t, testUsers)
    totalDistance += graphAdj[lastVisit, t] + stayTime[t]
    return (path, totalProfit, totalDistance)

def bestRatioPlusPath(s, t, testUsers):

    def sortRatio(edge):
        potDistance = totalDistance + edge[1] + stayTime[edge[0]]
        potProfit = totalProfit + evalFunc(edge[0], testUsers)
        if potDistance == 0:
            return float("inf")
        else:
            return potProfit/potDistance

    path=[s]
    totalProfit = evalFunc(s, testUsers)
    totalDistance = 0
    lastVisit = s
    oldPath = []
    swapping=False

    while True:
        oldPath = path.copy()
        graph[lastVisit].sort(key=sortRatio,reverse=True)
        for neigh in graph[lastVisit]:
            currProfit = evalFunc(neigh[0], testUsers)
            if ((currProfit > 0) and (neigh[0]!=t) and (neigh[0] not in path) and (
                    totalDistance + neigh[1] + stayTime[neigh[0]] + graphAdj[neigh[0], t] <= B)):
                path.append(neigh[0])
                totalProfit += currProfit
                lastVisit = neigh[0]
                totalDistance += neigh[1] + stayTime[neigh[0]]
                break
        #print(path,totalProfit/totalDistance)

        if path == oldPath:
            path.append(t)
            totalProfit += evalFunc(t, testUsers)
            totalDistance += graphAdj[lastVisit, t] + stayTime[t]
            break

    while True:
        notInPath=[]
        for x in range(1,len(graph)+1):
            if x not in path:
                notInPath.append(x)

        swaps=[]
        for i in range(1,len(path)-1):
            for j in notInPath:
                if (totalDistance - graphAdj[path[i-1],path[i]] - graphAdj[path[i],path[i+1]]
                        + graphAdj[path[i-1],j] + graphAdj[j,path[i+1]] - stayTime[path[i]] + stayTime[j]<= B):
                    swaps.append((i,j))

        maxRatio=totalProfit/totalDistance
        maxSwap=(0,0)

        for swap in swaps:
            extraProfit=evalFunc(swap[1], testUsers)-evalFunc(path[swap[0]], testUsers)
            extraDistance=-graphAdj[path[swap[0]-1],path[swap[0]]] \
                -graphAdj[path[swap[0]],path[swap[0]+1]]+graphAdj[path[swap[0]-1],swap[1]] \
                +graphAdj[swap[1],path[swap[0]+1]]-stayTime[path[swap[0]]]+stayTime[swap[1]]

            if (totalProfit+extraProfit)/(totalDistance+extraDistance)>maxRatio:
                maxRatio=(totalProfit+extraProfit)/(totalDistance+extraDistance)
                maxSwap=swap

        if maxSwap==(0,0):
            break
        else:
            #print("Path length is "+str(len(path))+".")
            #print("Making swap. Old profit is "+str(totalProfit)+".")
            totalProfit += evalFunc(maxSwap[1], testUsers) - evalFunc(path[maxSwap[0]], testUsers)
            totalDistance += graphAdj[path[maxSwap[0]-1],maxSwap[1]] + graphAdj[maxSwap[1],path[maxSwap[0]+1]] \
                            - graphAdj[path[maxSwap[0]-1],path[maxSwap[0]]] - graphAdj[path[maxSwap[0]],path[maxSwap[0]+1]] \
                            -stayTime[path[maxSwap[0]]]+stayTime[maxSwap[1]]
            path[maxSwap[0]] = maxSwap[1]
            #print("New profit is "+str(totalProfit)+".")
    #print("END")
    return (path, totalProfit, totalDistance)

def bestRatioPlusPlusPath(s, t, testUsers):

    def sortRatio(edge):
        potDistance = totalDistance + edge[1] + stayTime[edge[0]]
        potProfit = totalProfit + evalFunc(edge[0], testUsers)
        if potDistance == 0:
            return float("inf")
        else:
            return potProfit/potDistance

    path=[s]
    totalProfit = evalFunc(s, testUsers)
    totalDistance = 0
    lastVisit = s
    oldPath = []
    swapping=False

    while True:
        oldPath = path.copy()
        graph[lastVisit].sort(key=sortRatio,reverse=True)
        for neigh in graph[lastVisit]:
            currProfit = evalFunc(neigh[0], testUsers)
            if ((currProfit > 0) and (neigh[0]!=t) and (neigh[0] not in path) and (
                    totalDistance + neigh[1] + stayTime[neigh[0]] + graphAdj[neigh[0], t] <= B)):
                path.append(neigh[0])
                totalProfit += currProfit
                lastVisit = neigh[0]
                totalDistance += neigh[1] + stayTime[neigh[0]]
                break
        #print(path,totalProfit/totalDistance)

        if path == oldPath:
            path.append(t)
            totalProfit += evalFunc(t, testUsers)
            totalDistance += graphAdj[lastVisit, t] + stayTime[t]
            break

    while True:
        notInPath=[]
        for x in range(1,len(graph)+1):
            if x not in path:
                notInPath.append(x)

        swaps=[]
        for i in range(1,len(path)-1):
            for j in notInPath:
                for k in notInPath:
                    if (j!=k):
                        if (totalDistance - graphAdj[path[i-1],path[i]] - graphAdj[path[i],path[i+1]] \
                                + graphAdj[path[i-1],j] + graphAdj[j,path[i+1]] - graphAdj[path[-2],path[-1]] \
                                + graphAdj[path[-2],k] + graphAdj[k,path[-1]] - stayTime[path[i]] + stayTime[j] \
                                + stayTime[k] <= B):
                            swaps.append((i,j,k))

        maxRatio=totalProfit/totalDistance
        maxSwap=(0,0,0)

        for swap in swaps:
            extraProfit=evalFunc(swap[1], testUsers)+evalFunc(swap[2], testUsers)-evalFunc(path[swap[0]], testUsers)
            extraDistance=-graphAdj[path[swap[0]-1],path[swap[0]]] \
                -graphAdj[path[swap[0]],path[swap[0]+1]]+graphAdj[path[swap[0]-1],swap[1]] \
                +graphAdj[swap[1],path[swap[0]+1]]-graphAdj[path[-2],path[-1]]+graphAdj[path[-2],swap[2]] \
                +graphAdj[swap[2],path[-1]] - stayTime[path[swap[0]]] + stayTime[swap[1]] + stayTime[swap[2]]

            if (totalProfit+extraProfit)/(totalDistance+extraDistance)>maxRatio:
                maxRatio=(totalProfit+extraProfit)/(totalDistance+extraDistance)
                maxSwap=swap

        if maxSwap==(0,0,0):
            break
        else:
            #print("Path length is "+str(len(path))+".")
            #print("Making swap. Old profit is "+str(totalProfit)+".")
            totalProfit += evalFunc(maxSwap[1], testUsers)+evalFunc(maxSwap[2], testUsers) - evalFunc(path[maxSwap[0]], testUsers)
            totalDistance += graphAdj[path[maxSwap[0]-1],maxSwap[1]] + graphAdj[maxSwap[1],path[maxSwap[0]+1]] \
                            - graphAdj[path[maxSwap[0]-1],path[maxSwap[0]]] - graphAdj[path[maxSwap[0]],path[maxSwap[0]+1]] \
                            - graphAdj[path[-2],path[-1]]+graphAdj[path[-2],maxSwap[2]]+graphAdj[maxSwap[2],path[-1]] \
                            - stayTime[path[maxSwap[0]]] + stayTime[maxSwap[1]] + stayTime[maxSwap[2]]
            path[maxSwap[0]] = maxSwap[1]
            path[-1] = maxSwap[2]
            path.append(t)
            #print("New profit is "+str(totalProfit)+".")
    #print("END")
    return (path, totalProfit, totalDistance)

evalFunc = satisfactionSum
totalReps = 50
B = 420  # Budget (minutes)

while True:
    (s, t) = random.sample(range(1,len(graph)+1), 2)
    if graphAdj[s, t] <= B:
        break

with open('output.txt','w') as f:
    f.write("%d %d\n" % (s,t))

bestDistance = []
bestValue=[]
bestRatio=[]
bestRatioPlus=[]
bestRatioPlusPlus=[]
#bestBUMA=[]
for rep in range(totalReps):
    #print(rep)
    with open('output.txt','a') as f:
        f.write("%d\n" % rep)
    
    distanceScore = []
    valueScore = []
    ratioScore = []
    ratioPScore = []
    ratioPPScore = []
    #BUMAScore = []

    for k in range(1, 21):
        #print("K is "+str(k))
        testUsers = []
        testSet = random.sample(range(1,len(users)+1), k)
        for l in testSet:
            testUsers.append(users[l])
        testUsers = np.array([np.array(xi) for xi in testUsers])

        distanceScore.append(bestDistancePath(s, t, testUsers)[1])
        valueScore.append(bestValuePath(s, t, testUsers)[1])
        ratioScore.append(bestRatioPath(s, t, testUsers)[1])
        ratioPScore.append(bestRatioPlusPath(s, t, testUsers)[1])
        ratioPPScore.append(bestRatioPlusPlusPath(s, t, testUsers)[1])
        #BUMAScore.append(BUMAPath(s, t, testUsers)[1])
    #distanceScore[:]=[x/ratioScore[19] for x in distanceScore] #Normalization
    #valueScore[:]=[x/ratioScore[19] for x in valueScore] #Normalization
    #ratioScore[:]=[x/ratioScore[19] for x in ratioScore] #Normalization

    if rep == 0:
        bestDistance = distanceScore
        bestValue = valueScore
        bestRatio = ratioScore
        bestRatioPlus = ratioPScore
        bestRatioPlusPlus = ratioPPScore
        #bestBUMA = BUMAScore
    else:
        bestDistance = np.add(bestDistance, distanceScore)
        bestValue = np.add(bestValue, valueScore)
        bestRatio = np.add(bestRatio, ratioScore)
        bestRatioPlus = np.add(bestRatioPlus, ratioPScore)
        bestRatioPlusPlus = np.add(bestRatioPlusPlus, ratioPPScore)
        #bestBUMA = np.add(bestBUMA, BUMAScore)

bestDistance[:] = [x/totalReps for x in bestDistance]
bestValue[:] = [x/totalReps for x in bestValue]
bestRatio[:] = [x/totalReps for x in bestRatio]
bestRatioPlus[:] = [x/totalReps for x in bestRatioPlus]
bestRatioPlusPlus[:] = [x/totalReps for x in bestRatioPlusPlus]
#bestBUMA[:] = [x/totalReps for x in bestBUMA]

with open('bestValue.txt', 'w') as f1:
    for item in bestValue:
        f1.write("%f " % item)

with open('bestDistance.txt', 'w') as f1:
    for item in bestDistance:
        f1.write("%f " % item)

with open('bestRatio.txt', 'w') as f1:
    for item in bestRatio:
        f1.write("%f " % item)

with open('bestRatioPlus.txt', 'w') as f1:
    for item in bestRatioPlus:
        f1.write("%f " % item)

with open('bestRatioPlusPlus.txt', 'w') as f1:
    for item in bestRatioPlusPlus:
        f1.write("%f " % item)
