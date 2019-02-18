from groupTourLib import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys

city = "Rome"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

def heuristicsRep(rep):
    
    with open('log.txt','a') as f:
        f.write(str(rep)+'\n')
    
    while True:
        (s, t) = random.sample(range(1,len(graph)), 2)
        if pathCost([s,t],stayTime,graph) <= B:
            break

    #print(rep)
    distanceScore = []
    valueScore = []
    ratioScore = []
    ratioPScore = []
    # ratioPPScore = []

    distPath = bestDistancePath(s,t,graph,stayTime,B,pois)
    
    for k in klist:
        #print(rep,k)
        testUsers = []
        testSet = random.sample(list(users.keys()), k)
        for l in testSet:
            testUsers.append(users[l])

        distanceScore.append(pathProfit(distPath,testUsers,scoring,pois))
        
        valPath = bestValuePath(s, t, testUsers,graph,scoring,stayTime,B,pois)
        ratPath = bestRatioPath(s, t, testUsers,graph,scoring,stayTime,B,pois)
        ratPPath = bestRatioPlusPath(s, t, testUsers,graph,scoring,stayTime,B,pois)
        # ratPPPath = bestRatioPlusPlusPath(s,t,testUsers,graph,scoring,stayTime,B,pois)
        
        valueScore.append(pathProfit(valPath, testUsers, scoring,pois))
        ratioScore.append(pathProfit(ratPath, testUsers, scoring,pois))
        ratioPScore.append(pathProfit(ratPPath, testUsers, scoring,pois))
        # ratioPPScore.append(pathProfit(ratPPPath,testUsers,scoring,pois))

    return (distanceScore,valueScore,ratioScore,ratioPScore)

########## HEURISTICS SIMULATION & COMPARISON ##########
########################################################
########################################################
scoring = 'sum'
totalReps = 500
B = 300  # Budget (minutes)
klist=[1,2,5,10,20]

totalDistanceScore = []
totalValueScore=[]
totalRatioScore=[]
totalRatioPScore=[]
# totalRatioPPScore=[]
results=[]
numOfCores=int(sys.argv[1])

if numOfCores==1:
    for rep in range(totalReps):
        results.append(heuristicsRep(rep))
else:
    results = Parallel(n_jobs=numOfCores)(delayed(heuristicsRep)(rep) for rep in range(totalReps))

totalDistanceScore = [res[0] for res in results]
totalValueScore = [res[1] for res in results]
totalRatioScore = [res[2] for res in results]
totalRatioPScore = [res[3] for res in results]
# totalRatioPPScore = [res[4] for res in results]

with open('heuristics.dat','w') as f:
    f.write(str(totalDistanceScore))
    f.write(str(totalValueScore))
    f.write(str(totalRatioScore))
    f.write(str(totalRatioPScore))
    # f.write(str(totalRatioPPScore))

bestDistance = np.mean(totalDistanceScore,axis=0)
bestValue = np.mean(totalValueScore, axis=0)
bestRatio = np.mean(totalRatioScore, axis=0)
bestRatioPlus = np.mean(totalRatioPScore, axis=0)
# bestRatioPlusPlus = np.mean(totalRatioPPScore, axis=0)

plt.figure()
plt.plot(klist,bestDistance, marker='v', color='purple', linestyle='--')
plt.plot(klist,bestValue, marker='^', color='aqua', linestyle='--')
plt.plot(klist,bestRatio, marker='D', color='r', linestyle='--')
plt.plot(klist,bestRatioPlus, marker='s', color='fuchsia', linestyle='--')
# plt.plot(klist,bestRatioPlusPlus, marker='o', color='darkblue', linestyle='--')

plt.legend(['bestDistance', 'bestValue','bestRatio','bestRatio+'])
plt.xticks(klist)
plt.xlabel('Group size')
plt.ylabel('Average Solution Value')
plt.title('Satisfaction Sum')
plt.tight_layout()
plt.savefig('heuristics.png')
