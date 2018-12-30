from groupTourLib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

city = "Rome"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

########## USERS & POIS VARIANCE ##########
########################################### 
usersMean = np.mean(list(users.values()),axis=0)
usersVar=[]
for user in list(users.values()):
    usersVar.append((np.linalg.norm(np.subtract(user,usersMean)))**2)

poisMean = np.mean(list(pois.values()),axis=0)
poisVar=[]
for poi in list(pois.values()):
    poisVar.append((np.linalg.norm(np.subtract(poi,poisMean)))**2)

userVal,userBins=np.histogram(usersVar,bins=30)
userVal=[x/sum(userVal) for x in userVal]
poiVal,poiBins=np.histogram(poisVar,bins=30)
poiVal=[x/sum(poiVal) for x in poiVal]

plt.figure()
plt.bar(poiBins[:-1],poiVal,align='edge',width=np.diff(poiBins),facecolor='red',edgecolor='k',alpha=0.6)
plt.bar(userBins[:-1],userVal,align='edge',width=np.diff(userBins),facecolor='green',edgecolor='k',alpha=0.7)
plt.title('Users & POIs Variance (Rome)')
plt.legend(['POIs','Users'])
plt.savefig('Plots/Dataset/dataVariance_Rome.png')

########## PREFERENCE PAIRS DISTRIBUTION ##########
###################################################
preferenceList = []
for user in list(users.values()):
    for poi in list(pois.values()):
        preferenceList.append(np.dot(user,poi))

prefVal,prefBins=np.histogram(preferenceList,bins=50)
prefVal=[x/sum(prefVal) for x in prefVal]
plt.figure()
plt.bar(prefBins[:-1],prefVal,align='edge',width=np.diff(prefBins),edgecolor='k')
plt.title('User-POI Preference Distribution (Rome)')
plt.savefig('Plots/Dataset/preferenceDist_Rome.png')

########## USERS & POIS - PCA & VISUALIZATION ##########
########################################################
pca = PCA(n_components=2).fit(list(pois.values()))
reduced_data = pca.transform(list(pois.values()))
plt.figure()
plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
reduced_data = pca.transform(list(users.values()))
plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
plt.legend(['POIs','Users'])
plt.title('Users & POIs Visualization (Rome)')
plt.savefig('Plots/Dataset/PCA_Rome.png')