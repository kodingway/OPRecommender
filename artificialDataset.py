from groupTourLib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv

city = "Rome_Artificial_0"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

std=0.1
for userId in users:
    users[userId]=(np.add(users[userId],np.random.normal(np.zeros((1,8)),scale=std))).tolist()[0]

###################################
with open('Users_Profiles.txt','w') as csvFile:
    writer = csv.writer(csvFile,delimiter=' ',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for userId in users: 
        writer.writerow(list(users[userId]))
