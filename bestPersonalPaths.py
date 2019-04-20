import numpy as np
import matplotlib.pyplot as plt
from groupTourLib import *
import csv
import sys
from joblib import Parallel, delayed

city = "myFlorence"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

def pathRep(rep):
    possible=[userId for userId in users]
    userId = random.sample(possible,1)[0]
    with open('log.txt','a') as f:
        f.write(str(rep)+'\n')
    userPath=[]
    while(len(userPath)<=2):
        while True:
            (s, t) = random.sample(list(pois.keys()), 2)
            if pathCost([s,t],stayTime,graph) <= B:
                break
        userPath = bestRatioPlusPath(s, t, [users[userId]],graph,scoring,stayTime,B,pois)
    return(userPath)

B=420
scoring='sum'
numOfCores=int(sys.argv[1])
usersPaths=[]
totalReps=50
if numOfCores==1:    
    for rep in range(totalReps):
        usersPaths.append(pathRep(rep))
else:
    usersPaths = Parallel(n_jobs=numOfCores)(delayed(pathRep)(rep) for rep in range(totalReps))

with open('FlorencePersonalPaths.txt','w') as f:
    for path in usersPaths:
        for poiId in path:
            f.write(str(poiId)+' ')
        f.write('\n')
