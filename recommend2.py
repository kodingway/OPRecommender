import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from clusteringLib import *
from groupTourLib import *
import csv
from joblib import Parallel, delayed
import sys
import random

sourceCity='myRome'
targetCity='myPisa'
commonUsers=[]
with open('Dataset/Pisa+Rome.txt','r') as f:
    spamreader=csv.reader(f,delimiter=' ')
    for row in spamreader:
        commonUsers.append(row[0])

with open('Dataset/PisaPersonalPaths.txt','r') as f:
    spamreader=csv.reader(f,delimiter=' ')
    testPaths=[]
    for row in spamreader:
        myPath=[]
        for poi in row[:-1]:
            myPath.append(int(poi))
        testPaths.append(myPath)

graph={}
users={}
pois={}
stayTime={}
scoring='sum'
B=420

for city in [sourceCity,targetCity]:
    graph[city] = readGraph(city)
    users[city] = readUsers(city)
    pois[city] = readPOIs(city,len(graph[city]))
    stayTime[city] = readStayTimes(city,len(graph[city]))

popularityMatrix={}
with open('Dataset/'+targetCity+'/Popularity.txt','r') as f:
    spamreader=csv.reader(f,delimiter=' ')
    for row in spamreader:
        popularityMatrix[int(row[0])]=int(row[1])

for poiId in pois[targetCity]:
    if poiId not in popularityMatrix:
        popularityMatrix[poiId]=1

def pathSimilarity(sourcePath,targetPath,pois):
    similarity=0
    for targetPOI in targetPath:
        bestSim=-1
        for sourcePOI in sourcePath:
            # dist=np.linalg.norm(np.subtract(pois[targetCity][targetPOI],pois[sourceCity][sourcePOI]))
            # sim=1/(1+dist)
            sim=np.dot(pois[targetCity][targetPOI],pois[sourceCity][sourcePOI])
            if sim > bestSim:
                bestSim=sim
        similarity+=bestSim
    if len(targetPath)>2: 
        return(similarity)
        # return(similarity/(len(targetPath)-2))
    else:
        return(1)

def pathDistance(path1,path2,pois):
    distance1=0
    for poiI in path1:
        bestDist=100000
        for poiJ in path2:
            dist=np.linalg.norm(np.subtract(pois[targetCity][poiI],pois[targetCity][poiJ]))
            if dist < bestDist:
                bestDist=dist
        distance1+=bestDist
    if len(path1)>2:
        distance1/=(len(path1)-2)
    else:
        distance1=0

    distance2=0
    for poiI in path2:
        bestDist=100000
        for poiJ in path1:
            dist=np.linalg.norm(np.subtract(pois[targetCity][poiI],pois[targetCity][poiJ]))
            if dist < bestDist:
                bestDist=dist
        distance2+=bestDist
    if len(path2)>2:
        distance2/=(len(path2)-2)
    else:
        distance2=0
    
    return(max(distance1,distance2))

def merge_and_count(a, b):
    assert a == sorted(a,reverse=True) and b == sorted(b,reverse=True)
    c = []
    count = 0
    i, j = 0, 0
    while i < len(a) and j < len(b):
        c.append(max(b[j], a[i]))
        if b[j] > a[i]:
            count += len(a) - i # number of elements remaining in `a`
            j+=1
        else:
            i+=1
    # now we reached the end of one the lists
    c += a[i:] + b[j:] # append the remainder of the list to C
    return count, c

def sort_and_count(L):
    if len(L) == 1: return 0, L
    n = len(L) // 2 
    a, b = L[:n], L[n:]
    ra, a = sort_and_count(a)
    rb, b = sort_and_count(b)
    r, L = merge_and_count(a, b)
    return ra+rb+r, L

def computeAUC(realVec, predictedVec, paths, pois):
    paths.sort(reverse=True,key=lambda p: pathProfit(p, [predictedVec], scoring, pois[targetCity]))
    ranking = [pathProfit(p, [realVec], scoring, pois[targetCity]) for p in paths]
    swaps = sort_and_count(ranking)[0]
    AUC = 1-swaps/(len(ranking)*(len(ranking)-1)/2)
    return AUC
    
def popular(s,t,popularityMatrix):
    userTestPath=bestRatioPlusPathTABLE(s,t, graph[targetCity],scoring,stayTime[targetCity],B,pois[targetCity],popularityMatrix)
    return(userTestPath)

def randomCluster(s,t,clusterPaths):
    ######## New vector = random centroid ########
    chosenOne=random.randint(1,len(clusterPaths))
    userTestPath=clusterPaths[chosenOne][1]
    testUsers=[users[targetCity][userId] for userId in clusterPaths[chosenOne][0]]
    userTestVec=np.mean(testUsers,axis=0)
    return((userTestPath,userTestVec))

def nearestCluster(s,t,userId,users,clusterPaths,pois,userTrainPath):
    ######## New vector = centroid of closest cluster ########
    bestSimil=0
    for clusterId in clusterPaths:
        targetPath=clusterPaths[clusterId][1]
        simil=pathSimilarity(userTrainPath,targetPath,pois)
        if simil>bestSimil:
            bestSimil=simil
            bestCluster=clusterId
    testUsers=[users[targetCity][userId] for userId in clusterPaths[bestCluster][0]]
    userTestVec=np.mean(testUsers,axis=0)
    userTestPath=clusterPaths[bestCluster][1]
    return((userTestPath,userTestVec))

def weightedCluster(s,t,userId,users,clusterPaths,pois,userTrainPath):
    ######## New vector = weighted average of centroids ########
    userTestVec=np.zeros(10)
    accSimil=0
    for clusterId in clusterPaths:
        clusterUsers=[users[targetCity][userId] for userId in clusterPaths[clusterId][0]]
        centroid=np.mean(clusterUsers,axis=0)
        targetPath=clusterPaths[clusterId][1]
        simil=pathSimilarity(userTrainPath,targetPath,pois)
        accSimil+=simil
        userTestVec=np.add(userTestVec,[x*simil for x in centroid])
    userTestVec=[x/accSimil for x in userTestVec]
    userTestPath=bestRatioPlusPath(s, t, [userTestVec],graph[targetCity],scoring,stayTime[targetCity],B,pois[targetCity])
    return((userTestPath,userTestVec))

def linearComb(s,t,userId,users,pois,userTrainVec,alpha):
    targetUsers=[users[targetCity][myId] for myId in users[targetCity] if myId not in commonUsers]
    targetMean=np.mean(targetUsers,axis=0)
    userTestVec=[alpha*x+(1-alpha)*y for (x,y) in zip(userTrainVec,targetMean)]
    userTestPath=bestRatioPlusPath(s, t, [userTestVec],graph[targetCity],scoring,stayTime[targetCity],B,pois[targetCity])
    return((userTestPath,userTestVec))

def itemCollaborative(s,t,userId,users,pois,userTrainVec,alpha):
    ######## ???????????????????? ########
    # realPreference=[]
    # for poiId in pois[targetCity]:
    #     realPreference.append(np.dot(users[targetCity][userId],pois[targetCity][poiId]))
    # prefVal,prefBins=np.histogram(realPreference,bins=50)
    # prefVal=[x/sum(prefVal) for x in prefVal]
    # # plt.figure()
    # # plt.bar(prefBins[:-1],prefVal,align='edge',width=np.diff(prefBins),edgecolor='k')
    # # plt.title('User-POI preference distribution')
    
    # table={}
    # for poiI in pois[targetCity]:
    #     table[poiI]=0
    #     accSimil=0
    #     for poiJ in pois[sourceCity]:
    #         simil = np.dot(pois[targetCity][poiI],pois[sourceCity][poiJ])
    #         score = np.dot(pois[sourceCity][poiJ],userTrainVec)
    #         table[poiI]+=score*(100**(10*simil))
            # accSimil+=(100**(10*simil))
            # table[poiI]+=score*simil
            # accSimil+=simil
        # table[poiI]/=accSimil
    # prefVal,prefBins=np.histogram(list(table.values()),bins=50)
    # prefVal=[x/sum(prefVal) for x in prefVal]
    # plt.bar(prefBins[:-1],prefVal,align='edge',width=np.diff(prefBins),edgecolor='k')
    # plt.show()
    targetUsers=[users[targetCity][myId] for myId in users[targetCity] if myId not in commonUsers]
    targetMean=np.mean(targetUsers,axis=0)
    table={}
    for poiI in pois[targetCity]:
        table[poiI]=alpha*np.dot(pois[targetCity][poiI],userTrainVec)+(1-alpha)*np.dot(pois[targetCity][poiI],targetMean)
    AMatrix=[]
    bMatrix=[]
    for poiId in pois[targetCity]:
        AMatrix.append(pois[targetCity][poiId])
        bMatrix.append(table[poiId])

    AMatrix=np.array(AMatrix)
    AMatrix=np.matmul(AMatrix,np.transpose(AMatrix))
    bMatrix=np.array(bMatrix)
    multipliers=list(nnls(AMatrix,bMatrix)[0])
    # multipliers=list(np.linalg.lstsq(AMatrix,bMatrix,rcond=None)[0])
    userTestVec=np.sum([[x*multipliers[poiId-1] for x in pois[targetCity][poiId]] for poiId in pois[targetCity]],axis=0)
    userTestVec=[x/sum(multipliers) for x in userTestVec]
    userTestPath=bestRatioPlusPathTABLE(s,t, graph[targetCity],scoring,stayTime[targetCity],B,pois[targetCity],table)
    return((userTestPath,userTestVec))

def recomRep(rep,userId):

    userTrainVec=users[sourceCity][userId]
    userRealVec=users[targetCity][userId]
    with open('log.txt','a') as f:
        f.write(str(rep)+'\n')
    
    # plt.figure()
    # legend=[]
    # reduced_data = pca.transform(list(pois[sourceCity].values()))
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
    # legend.append('Source POIs')
    # reduced_data = pca.transform(list(pois[targetCity].values()))
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
    # legend.append('Targets POIs')
    # reduced_data = pca.transform([userTrainVec])
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='o')
    # legend.append('Source User')
    # reduced_data = pca.transform([userRealVec])
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='o')
    # legend.append('Target User')

    while True:
        (s, t) = random.sample(list(pois[sourceCity].keys()), 2)
        if pathCost([s,t],stayTime[sourceCity],graph[sourceCity]) <= B:
            break
    
    userTrainPath=bestRatioPlusPath(s, t, [userTrainVec],graph[sourceCity],scoring,stayTime[sourceCity],B,pois[sourceCity])
    
    while True:
        (s, t) = random.sample(list(pois[targetCity].keys()), 2)
        if pathCost([s,t],stayTime[targetCity],graph[targetCity]) <= B:
            break

    # clusterPaths={}
    # with open('Dataset/'+targetCity[2:]+'Clusters'+'.txt','r') as f:
    #     spamreader=csv.reader(f, delimiter=';')
    #     myId=1
    #     for row in spamreader:
    #         clusterUsers=list(row[0].strip().split(' '))
    #         testUsers=[users[targetCity][userId] for userId in clusterUsers if userId not in commonUsers]
    #         clusterPath=bestRatioPlusPath(s, t, [np.mean(testUsers,axis=0)],graph[targetCity],scoring,stayTime[targetCity],B,pois[targetCity])
    #         clusterPaths[myId]=(clusterUsers,clusterPath)
    #         myId+=1


    # popularPath=popular(s,t,popularityMatrix)
    # popularProfit=pathProfit(popularPath, [userRealVec], scoring, pois[targetCity])
    
    # randPath,randVec=randomCluster(s,t,clusterPaths)
    # randProfit=pathProfit(randPath, [userRealVec], scoring, pois[targetCity])
    # reduced_data = pca.transform([randVec])
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='D')
    # legend.append('Random')

    # nearestPath,nearestVec=nearestCluster(s,t,userId,users,clusterPaths,pois,userTrainPath)
    # nearestProfit=pathProfit(nearestPath, [userRealVec], scoring, pois[targetCity])
    # reduced_data = pca.transform([nearestVec])
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='X')
    # legend.append('Nearest')

    # weightedPath,weightedVec=weightedCluster(s,t,userId,users,clusterPaths,pois,userTrainPath)
    # weightedProfit=pathProfit(weightedPath, [userRealVec], scoring, pois[targetCity])
    # reduced_data = pca.transform([weightedVec])
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='P')
    # legend.append('Weighted')

    # globalAveragePath,globalAverageVec=linearComb(s,t,userId,users,pois,userTrainVec,0)
    # globalAverageProfit=pathProfit(globalAveragePath,[userRealVec],scoring,pois[targetCity])
    
    collabList=[]
    for alpha in alphaList:
        collabPath,collabVec=linearComb(s,t,userId,users,pois,userTrainVec,alpha)
        collabProfit=pathProfit(collabPath,[userRealVec],scoring,pois[targetCity])
        collabList.append((collabProfit,collabPath,collabVec))

    # plt.legend(legend)
    # plt.show()

    userOptimalPath=bestRatioPlusPath(s, t, [userRealVec],graph[targetCity],scoring,stayTime[targetCity],B,pois[targetCity])
    userOptimalProfit=pathProfit(userOptimalPath, [userRealVec], scoring, pois[targetCity])


    if userOptimalProfit!=0:
        # popularScore=min(1,popularProfit/userOptimalProfit)
        # globalAverageScore=min(1,globalAverageProfit/userOptimalProfit)
        # randScore=min(1,randProfit/userOptimalProfit)
        # nearestScore=min(1,nearestProfit/userOptimalProfit)
        # weightedScore=min(1,weightedProfit/userOptimalProfit)
        collabScores=[min(1,profit/userOptimalProfit) for profit,_,_ in collabList]
    else:
        # popularScore,globalAverageScore,randScore,nearestScore,weightedScore=(0,0,0,0,0)
        collabScores=np.zeros(len(collabList))

    collabSetDist=[pathDistance(userOptimalPath,path,pois) for _,path,_ in collabList]
    # randSetDist=pathDistance(userOptimalPath,randPath,pois)
    # globalAverageSetDist=pathDistance(userOptimalPath,globalAveragePath,pois)
    # nearestSetDist=pathDistance(userOptimalPath,nearestPath,pois)
    # weightedSetDist=pathDistance(userOptimalPath,weightedPath,pois)
    
    collabVecDist=[np.linalg.norm(np.subtract(userRealVec,vec)) for _,_,vec in collabList]
    # randVecDist=np.linalg.norm(np.subtract(userRealVec,randVec))
    # globalAverageVecDist=np.linalg.norm(np.subtract(userRealVec,globalAverageVec))
    # nearestVecDist=np.linalg.norm(np.subtract(userRealVec,nearestVec))
    # weightedVecDist=np.linalg.norm(np.subtract(userRealVec,weightedVec))

    collabAUC = [computeAUC(userRealVec,vec,testPaths,pois) for _,_,vec in collabList] 
    # randAUC=computeAUC(userRealVec,randVec,testPaths,pois)
    # globalAverageAUC=computeAUC(userRealVec,globalAverageVec,testPaths,pois)
    # nearestAUC=computeAUC(userRealVec,nearestVec,testPaths,pois)
    # weightedAUC=computeAUC(userRealVec,weightedVec,testPaths,pois)

    # rand=(randScore,randSetDist,randVecDist,randAUC)
    # globalAverage=(globalAverageScore,globalAverageSetDist,globalAverageVecDist,globalAverageAUC)
    # nearest=(nearestScore,nearestSetDist,nearestVecDist,nearestAUC)
    # weighted=(weightedScore,weightedSetDist,weightedVecDist,weightedAUC)

    # return((popularScore,rand,nearest,weighted,globalAverage))
    return(collabScores,collabSetDist,collabVecDist,collabAUC)

numOfCores=int(sys.argv[1])
results=[]
alphaList=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
# pca=PCA(n_components=2).fit(list(pois[sourceCity].values())+list(pois[targetCity].values()))
if numOfCores==1:
    for rep,userId in enumerate(commonUsers):
        results.append(recomRep(rep,userId))
else:
    results = Parallel(n_jobs=numOfCores)(delayed(recomRep)(rep,userId) for rep,userId in enumerate(commonUsers))

collabScores=[res[0] for res in results]
collabSetDist=[res[1] for res in results]
collabVecDist=[res[2] for res in results]
collabAUC=[res[3] for res in results]
# popularScores = [res[0] for res in results]
# randomResults = [res[1] for res in results]
# randomScores=[res[0] for res in randomResults]
# randomSetDist=[res[1] for res in randomResults]
# randomVecDist=[res[2] for res in randomResults]
# randomAUC=[res[3] for res in randomResults]

# nearestResults = [res[2] for res in results]
# nearestScores=[res[0] for res in nearestResults]
# nearestSetDist=[res[1] for res in nearestResults]
# nearestVecDist=[res[2] for res in nearestResults]
# nearestAUC=[res[3] for res in nearestResults]

# weightedResults = [res[3] for res in results]
# weightedScores=[res[0] for res in weightedResults]
# weightedSetDist=[res[1] for res in weightedResults]
# weightedVecDist=[res[2] for res in weightedResults]
# weightedAUC=[res[3] for res in weightedResults]

# globalResults=[res[4] for res in results]
# globalScores=[res[0] for res in globalResults]
# globalSetDist=[res[1] for res in globalResults]
# globalVecDist=[res[2] for res in globalResults]
# globalAUC=[res[3] for res in globalResults]

directory='Algos/RP/'
with open(directory+'collabScores.txt','w') as f:
    f.write(str(collabScores))
with open(directory+'collabSetDist.txt','w') as f:
    f.write(str(collabSetDist))
with open(directory+'collabVecDist.txt','w') as f:
    f.write(str(collabVecDist))
with open(directory+'collabAUC.txt','w') as f:
    f.write(str(collabAUC))
# with open(directory+'popularScores.txt','w') as f:
#     f.write(str(popularScores))
# with open(directory+'randomScores.txt','w') as f:
#     f.write(str(randomScores))
# with open(directory+'globalScores.txt','w') as f:
#     f.write(str(globalScores))
# with open(directory+'nearestScores.txt','w') as f:
#     f.write(str(nearestScores))
# with open(directory+'weightedScores.txt','w') as f:
#     f.write(str(weightedScores))

# with open(directory+'randomSetDist.txt','w') as f:
#     f.write(str(randomSetDist))
# with open(directory+'globalSetDist.txt','w') as f:
#     f.write(str(globalSetDist))
# with open(directory+'nearestSetDist.txt','w') as f:
#     f.write(str(nearestSetDist))
# with open(directory+'weightedSetDist.txt','w') as f:
#     f.write(str(weightedSetDist))

# with open(directory+'randomVecDist.txt','w') as f:
#     f.write(str(randomVecDist))
# with open(directory+'globalVecDist.txt','w') as f:
#     f.write(str(globalVecDist))
# with open(directory+'nearestVecDist.txt','w') as f:
#     f.write(str(nearestVecDist))
# with open(directory+'weightedVecDist.txt','w') as f:
#     f.write(str(weightedVecDist))

# with open(directory+'randomAUC.txt','w') as f:
#     f.write(str(randomAUC))
# with open(directory+'globalAUC.txt','w') as f:
#     f.write(str(globalAUC))
# with open(directory+'nearestAUC.txt','w') as f:
#     f.write(str(nearestAUC))
# with open(directory+'weightedAUC.txt','w') as f:
#     f.write(str(weightedAUC))