from groupTourLib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
from clusteringLib import *

city = "myPisa"
graph = readGraph(city)
pois = readPOIs(city,len(graph))

clusterIds=kmeans(4,list(pois.keys()),pois,init='kmeans++')

pca = PCA(n_components=2).fit(list(pois.values()))
plt.figure()
for ind,cluster in enumerate(clusterIds):
    reduced_data = pca.transform([pois[clusterId] for clusterId in cluster])
    plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10,marker='D')
    plt.text(reduced_data[0,0],reduced_data[0,1],str(ind),color='k',fontsize=10)
plt.show()

syntheticUsers=[]
for poiId in clusterIds[1]:
    syntheticUsers.append(pois[poiId])

std=0.01
with open('Dataset/'+city+'/Users_Profiles.txt','r') as fr:
    with open('Users_Profiles.txt','w') as fw:
        spamreader=csv.reader(fr,delimiter=' ')
        for row in spamreader:
            userId = row[0]
            userProfile = random.sample(syntheticUsers,1)[0]
            userProfile = np.add(userProfile,np.random.normal(np.zeros((1,10)),scale=std)).tolist()[0]  
            fw.write(userId+' ')
            for num in userProfile:
                fw.write(str(num)+' ')
            fw.write('\n')