from groupTourLib import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
    # ratioPScore = []
    # ratioPPScore = []

    # print("Best distance")
    distPath = bestDistancePath(s,t,graph,stayTime,B)
    
    for k in klist:
        #print(rep,k)
        testUsers = []
        testSet = random.sample(range(1,len(users)+1), k)
        for l in testSet:
            testUsers.append(users[l])
        testUsers = np.array([np.array(xi) for xi in testUsers])

        distanceScore.append(pathProfit(distPath,testUsers,scoreFunc,pois))
        # print("Best value")
        valPath = bestValuePath(s, t, testUsers,graph,scoreFunc,stayTime,B,pois)
        # print("Best ratio")
        ratPath = bestRatioPath(s, t, testUsers,graph,scoreFunc,stayTime,B,pois)
        # print("Best ratio+")
        # ratPPath = bestRatioPlusPath(s, t, testUsers,graph,scoreFunc,stayTime,B,pois)
        # print("Best ratio++")
        # ratPPPath = bestRatioPlusPlusPath(s,t,testUsers,graph,scoreFunc,stayTime,B,pois)
        
        valueScore.append(pathProfit(valPath, testUsers, scoreFunc,pois))
        ratioScore.append(pathProfit(ratPath, testUsers, scoreFunc,pois))
        # ratioPScore.append(pathProfit(ratPPath, testUsers, scoreFunc,pois))
        # ratioPPScore.append(pathProfit(ratPPPath,testUsers,scoreFunc,pois))

    return (distanceScore,valueScore,ratioScore)

########## HEURISTICS SIMULATION & COMPARISON ##########
########################################################
########################################################
scoreFunc = satisfactionSum
totalReps = 500
B = 420  # Budget (minutes)
klist=[1,2,5,10,20]

totalDistanceScore = []
totalValueScore=[]
totalRatioScore=[]
totalRatioPScore=[]
# totalRatioPPScore=[]

results  = Parallel(n_jobs=15)(delayed(heuristicsRep)(rep) for rep in range(totalReps))
totalDistanceScore = [res[0] for res in results]
totalValueScore = [res[1] for res in results]
totalRatioScore = [res[2] for res in results]

bestDistance = np.mean(totalDistanceScore,axis=0)
bestValue = np.mean(totalValueScore, axis=0)
bestRatio = np.mean(totalRatioScore, axis=0)
# bestRatioPlus = np.mean(totalRatioPScore, axis=0)
# bestRatioPlusPlus = np.mean(totalRatioPPScore, axis=0)

with open('heuristics.dat','w') as f:
    f.write(str(bestDistance))
    f.write(str(bestValue))
    f.write(str(bestRatio))
    # f.write(bestRatioPlus)
    # f.write(bestRatioPlusPlus)

plt.figure()
plt.plot(klist,bestDistance, marker='v', color='purple', linestyle='--')
plt.plot(klist,bestValue, marker='^', color='aqua', linestyle='--')
plt.plot(klist,bestRatio, marker='D', color='r', linestyle='--')
# plt.plot(klist,bestRatioPlus, marker='s', color='fuchsia', linestyle='--')
# plt.plot(klist,bestRatioPlusPlus, marker='o', color='darkblue', linestyle='--')
plt.legend(['bestDistance', 'bestValue','bestRatio'])
plt.xticks(klist)
plt.xlabel('Group size')
plt.ylabel('Average Solution Value')
plt.title('Satisfaction Sum')
plt.tight_layout()
plt.savefig('heuristics.png')
