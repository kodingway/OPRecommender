import numpy as np
import csv
import random

def readGraph(city):
    with open("Dataset/" + city + "/Distances.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        walkSpeed = 10/60  # (km/min)
        
        count = 1
        for row in csvReader:
            if int(row[0])==1:
                count+=1
            else:
                break

        graph = np.zeros([count+1, count+1])
        csvFile.seek(0)
        for row in csvReader:
            graph[int(row[0]), int(row[1])] = float(row[2])/walkSpeed

    return graph

def readUsers(city):
    with open("Dataset/" + city + "/Users_Profiles.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        users = {}
        userId = 1
        for row in csvReader:
            users[userId]=list(map(float,row[:8]))
            userId += 1
    
    return users

def readPOIs(city,graphSize):
    with open("Dataset/" + city + "/POIs_Vectors.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        pois = {}
        for row in csvReader:
            pois[int(row[0])]=list(map(float,row[1:9]))

    for key in range(1,graphSize):
        if key not in list(pois.keys()):
            pois[key]=pois[1]  #some pois don't have vectors

    return pois

def readStayTimes(city,graphSize):
    with open("Dataset/" + city + "/POIs_StayTime.txt") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=' ')

        stayTime={}
        for row in csvReader:
            stayTime[int(row[0])] = float(row[1])/60 # (min)

    stayMax=max(stayTime, key=stayTime.get)
    stayMin=min(stayTime, key=stayTime.get)
    for key in range(1,graphSize):
        if key not in list(stayTime.keys()):
            stayTime[key]=random.uniform(stayTime[stayMin],stayTime[stayMax]) #a lot of pois don't have stay times
    return stayTime

def satisfactionSum(poiId, testUsers, pois):
    return sum(np.matmul(testUsers, pois[poiId]))

def satisfactionMin(poiId, testUsers, pois):
    return min(np.matmul(testUsers, pois[poiId]))

def satisfactionFair(poiId, testUsers, pois):
    tempSat=np.matmul(testUsers, pois[poiId])
    return np.mean(tempSat) - 0.5 * np.std(tempSat)

def pathCost(path, stayTime, graph):
    result=0
    for i,node in enumerate(path[0:-1]):
        result+=stayTime[node]+graph[node,path[i+1]]
    return(result)

def pathProfit(path,testUsers, scoreFunc,pois):
    result=0
    for node in path[1:-1]:
        result+=scoreFunc(node,testUsers,pois)
    return(result)

########## BEST VALUE HEURISTIC ##########
##########################################
def bestValuePath(s, t, testUsers, graph, scoreFunc, stayTime, B, pois):

    path=[s,t]
    while True:
        bestPath = path[:]
        bestValue = 0
        for neigh in range(1,len(graph)):
            if neigh not in path:
                potPath = path[:]
                potPath.insert(-1,neigh)
                if (pathProfit(potPath,testUsers,scoreFunc,pois)>bestValue) and (pathCost(potPath,stayTime,graph) <= B):
                    bestPath = potPath[:]
                    bestValue = pathProfit(bestPath,testUsers,scoreFunc,pois)
        if bestPath == path:
            return path
        path = bestPath[:]

########## BEST DISTANCE HEURISTIC ##########
#############################################
def bestDistancePath(s, t, graph, stayTime, B):

    path=[s,t]
    while True:
        closestDistance=float("inf")
        bestPath = path[:]
        for neigh in range(1,len(graph)):
            if neigh not in path:
                if (graph[path[-2],neigh]+stayTime[neigh]<closestDistance):
                    potPath=path[:]
                    potPath.insert(-1,neigh)
                    if (pathCost(potPath,stayTime,graph)<=B):
                        closestDistance=graph[path[-2],neigh]+stayTime[neigh]
                        bestPath = potPath[:]
        if bestPath == path:
            return path
        path = bestPath[:]

########## BEST RATIO HEURISTIC ##########
##########################################
def bestRatioPath(s, t, testUsers, graph, scoreFunc, stayTime, B, pois):

    path=[s,t]
    while True:
        changedPath = False
        bestPath = path[:]
        bestProfit=pathProfit(bestPath,testUsers,scoreFunc,pois)
        bestCost=pathCost(bestPath,stayTime,graph)
        for neigh in range(1,len(graph)):
            if neigh not in path:
                potPath = path[:]
                potPath.insert(-1,neigh)
                potProfit=pathProfit(potPath,testUsers,scoreFunc,pois)
                potCost=pathCost(potPath,stayTime,graph)
                if (potProfit/potCost>bestProfit/bestCost) and (potCost <= B):
                    changedPath=True
                    bestPath = potPath[:]
                    bestProfit=potProfit
                    bestCost=potCost
        if changedPath==False:
            return path
        path = bestPath[:]


########## BEST RATIO+ HEURISTIC ##########
###########################################
def bestRatioPlusPath(s, t, testUsers, graph, scoreFunc, stayTime, B,pois):

    path=[s,t]
    while True:
        changedPath=False
        bestPath = path[:]
        bestProfit=pathProfit(bestPath,testUsers,scoreFunc,pois)
        bestCost=pathCost(bestPath,stayTime,graph)
        for neigh in range(1,len(graph)):
            if neigh not in path:
                potPath = path[:]
                potPath.insert(-1,neigh)
                potProfit = pathProfit(potPath,testUsers,scoreFunc,pois)
                potCost = pathCost(potPath,stayTime,graph)
                if (potProfit/potCost>bestProfit/bestCost) and (potCost <= B):
                    changedPath=True
                    bestPath = potPath[:]
                    bestProfit=potProfit
                    bestCost=potCost
        if changedPath==False:
            break
        path = bestPath[:]

    while True:
        changedPath=False
        bestPath = path[:]
        bestProfit=pathProfit(bestPath,testUsers,scoreFunc,pois)
        bestCost=pathCost(bestPath,stayTime,graph)
        for neigh in range(1,len(graph)):
            if neigh not in path:
                for nodeI in range(1,len(path)-1):
                    potPath = path[:]
                    potPath[nodeI] = neigh
                    potProfit=pathProfit(potPath,testUsers,scoreFunc,pois)                   
                    potCost=pathCost(potPath,stayTime,graph)
                    if (potProfit/potCost>bestProfit/bestCost) and (potProfit>bestProfit) and (potCost <= B):
                        changedPath=True
                        bestPath = potPath[:]
                        bestProfit=potProfit
                        bestCost=potCost       
        if changedPath==False:
            return path
        path = bestPath[:]


########## BEST RATIO++ HEURISTIC ##########
############################################
def bestRatioPlusPlusPath(s, t, testUsers, graph, scoreFunc, stayTime, B,pois):

    path=[s,t]
    while True:
        changedPath=False
        bestPath = path[:]
        bestProfit=pathProfit(bestPath,testUsers,scoreFunc,pois)
        bestCost=pathCost(bestPath,stayTime,graph)
        for neigh in range(1,len(graph)):
            if neigh not in path:
                potPath = path[:]
                potPath.insert(-1,neigh)
                potProfit=pathProfit(potPath,testUsers,scoreFunc,pois)                   
                potCost=pathCost(potPath,stayTime,graph)
                if (potProfit/potCost>bestProfit/bestCost) and (potProfit>bestProfit) and (potCost <= B):
                    changedPath=True
                    bestPath = potPath[:]
                    bestProfit=potProfit
                    bestCost=potCost
        if changedPath==False:
            break
        path = bestPath[:]

    while True:
        changedPath=False
        bestPath = path[:]
        bestProfit=pathProfit(bestPath,testUsers,scoreFunc,pois)
        bestCost=pathCost(bestPath,stayTime,graph)
        for neigh in range(1,len(graph)):
            if neigh not in path:
                for nodeI in range(1,len(path)-1):
                    potPath = path[:]
                    potPath[nodeI] = neigh
                    potProfit=pathProfit(potPath,testUsers,scoreFunc,pois)                   
                    potCost=pathCost(potPath,stayTime,graph)
                    if (pathProfit(potPath,testUsers,scoreFunc,pois)/pathCost(potPath,stayTime,graph)>pathProfit(bestPath,testUsers,scoreFunc,pois)/pathCost(bestPath,stayTime,graph)) and (pathProfit(potPath,testUsers,scoreFunc,pois)>pathProfit(bestPath,testUsers,scoreFunc,pois)) and (pathCost(potPath,stayTime,graph) <= B):
                        changedPath=True
                        bestPath = potPath[:]
                        bestProfit=potProfit
                        bestCost=potCost
        if changedPath==False:
            break
        path = bestPath[:]

    while True:
        changedPath=False
        bestPath = path[:]
        bestProfit=pathProfit(bestPath,testUsers,scoreFunc,pois)
        bestCost=pathCost(bestPath,stayTime,graph)
        notInPath = []
        for neigh in range(1,len(graph)):
            if neigh not in path:
                notInPath.append(neigh)
        for neigh in notInPath:
                for neigh2 in notInPath:
                    if (neigh!=neigh2):
                        for nodeI in range(1,len(path)-1):
                            potPath = path[:]
                            potPath[nodeI] = neigh
                            potPath.insert(-1,neigh2)
                            potProfit=pathProfit(potPath,testUsers,scoreFunc,pois)                   
                            potCost=pathCost(potPath,stayTime,graph)
                            if (potProfit/potCost>bestProfit/bestCost) and (potProfit>bestProfit) and (potCost <= B):
                                changedPath=True
                                bestPath = potPath[:]
                                bestProfit=potProfit
                                bestCost=potCost
        if changedPath==False:
            return path
        path = bestPath[:]
