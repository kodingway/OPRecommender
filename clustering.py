import random
import numpy as np
from groupTourLib import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed
from clusteringLib import *

city = "Rome"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))
    
m=100 #Number of users
totalScores = []
B=420
scoring = 'sum'
totalReps=10
klist=[1,2,3,4,5,10,20,50]
########## VISUALIZATION ##########
# pca = PCA(n_components=2).fit(list(users.values()))
###################################

def clusteringRep(rep):
    with open('log.txt','a') as f:
        f.write(str(rep)+'\n')
    
    while True:
        (s, t) = random.sample(range(1,len(graph)), 2)
        if pathCost([s,t],stayTime,graph) <= B:
            break
    
    testSet = random.sample(range(1,len(users)+1), m)
    kscores=[]
    for k in klist:
        # print(rep,k)
        clusterIds = kmeans(k,testSet,users,init='kmeans++')
        meanList, varList = clusterMetrics(clusterIds,users)
        # print(np.sum(varList,axis=1))
        ########## VISUALIZATION ##########
        # reduced_data = pca.transform(list(pois.values()))
        # plt.figure()
        # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
        # reduced_data = pca.transform([pois[s],pois[t]])
        # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='P')
        # for ind,cluster in enumerate(clusterIds):
        #     reduced_data = pca.transform([users[clusterId] for clusterId in cluster])
        #     plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10,marker='D')
        #     reduced_data = pca.transform([meanList[ind]])
        #     plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='x',c='k')
        #     plt.show()
        ###################################
        clusterProfits=[]
        # plt.figure()
        for ind,cluster in enumerate(clusterIds):
            # reduced_data = pca.transform(list(pois.values()))
            # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
            # reduced_data = pca.transform([users[clusterId] for clusterId in cluster])
            # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10,marker='D')
            testUsers=[]
            for l in cluster:
                testUsers.append(users[l])
            clusterPath = bestRatioPlusPath(s, t, testUsers,graph,scoring,stayTime,B,pois)
            # reduced_data = pca.transform([pois[poiId] for poiId in clusterPath])
            # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='P')
            # for ind,dat in enumerate(reduced_data):
            #     plt.text(dat[0],dat[1],str(ind),color='k',fontsize=10)
            # plt.show()
            clusterProfits.append(pathProfit(clusterPath, testUsers, scoring, pois))
            # plt.clf()
        kscores.append(sum(clusterProfits))
    return kscores

numOfCores=int(sys.argv[1])
if numOfCores==1:
    for rep in range(totalReps):
        totalScores.append(clusteringRep(rep))
else:
    totalScores = Parallel(n_jobs=numOfCores)(delayed(clusteringRep)(rep) for rep in range(totalReps))

with open('clustering.dat','w') as f:
    f.write(str(totalScores))

totalScores=np.mean(totalScores,axis=0)

plt.figure()
plt.plot(klist,totalScores, marker='D', linestyle='--')
plt.xticks(klist)
plt.xlabel('Number of clusters')
plt.ylabel('Average solution value')
plt.title('Overall Group Satisfaction')
plt.tight_layout()
plt.savefig('heuristics.png')