import csv
import wikipedia
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim import corpora, models
import nltk
from math import sin, cos, sqrt, atan2, radians
import flickrapi
import time
import datetime

def coorDistance(lat1,lon1,lat2,lon2):
    latI=radians(lat1)
    lonI=radians(lon1)
    latJ=radians(lat2)
    lonJ=radians(lon2)
    dlon = lonJ - lonI
    dlat = latJ - latI
    a = sin(dlat/2)**2 + cos(latI)*cos(latJ)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R=6373.0
    return R*c

def findClosestPOI(myLat,myLon,pois):
    minDist=300000
    closestId=-1
    for poiId in pois:
        poiLat=pois[poiId]['lat']
        poiLon=pois[poiId]['lon']
        dist=coorDistance(myLat,myLon,poiLat,poiLon)
        if dist<minDist and dist<0.2:   #Closest POI in radius 200m
              minDist=dist
              closestId=poiId
    return closestId


centerName='Rome'
page=wikipedia.page(centerName)
centerLat=float(page.coordinates[0])
centerLon=float(page.coordinates[1])

api_key='8d1a7267a8e07b28c8997b8bf7ccabe8'
api_secret='66a3513acdb609b2'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
photos = flickr.photos.search(min_taken_date='2015-01-01',accuracy=16,lat=centerLat,lon=centerLon,radius=10,in_gallery=True)
numOfPages = photos['photos']['pages']

#GET ALL PHOTOS
photoDict=[]

for page in range(1,numOfPages+1):
    print(page)
    try:
        photos = flickr.photos.search(min_taken_date='2015-01-01', page=page,accuracy=16,lat=centerLat,lon=centerLon,radius=10,in_gallery=True)
        for photo in photos['photos']['photo']:
                photoDict.append(photo)
    except:
        continue

pois={}
rawTexts={}
poiTitleToId={}
myId=1

#GET ALL POIS
wikiRes=wikipedia.geosearch(centerLat,centerLon,results=1000,radius=10000)
for ind,title in enumerate(wikiRes):
    print(ind)
    try:
        page=wikipedia.page(title)
        plat,plon=map(float,page.coordinates)
        poiTitleToId[title]=myId
        pois[myId]={'name':title,'lat':plat,'lon':plon}
        rawTexts[myId]=page.content
        myId+=1
    except:
        continue

with open('dictionary.txt','w') as f:
    for title in poiTitleToId:
        f.write(title+'\t'+str(poiTitleToId[title])+'\t'+str(pois[poiTitleToId[title]]['lat'])+' '+str(pois[poiTitleToId[title]]['lon'])+'\n')

for rawId in rawTexts:
    with open('rawTexts/'+str(rawId)+'.txt','w') as f:
        f.write(str(rawTexts[rawId])+'\n')

with open('Distances.txt','w') as f:
    for poiI in pois:
        for poiJ in pois:
            dist=coorDistance(pois[poiI]['lat'],pois[poiI]['lon'],pois[poiJ]['lat'],pois[poiJ]['lon'])
            f.write(str(poiI)+' '+str(poiJ)+' '+str(dist)+'\n')

users={}

for ind,photo in enumerate(photoDict):
    print(ind)
    try:
        info = flickr.photos.getInfo(api_key=api_key,photo_id=photo['id'])
        owner=info['photo']['owner']['nsid']
        lat=float(info['photo']['location']['latitude'])
        lon=float(info['photo']['location']['longitude'])
        timeTaken=info['photo']['dates']['taken']
        timeStamp=int(time.mktime(datetime.datetime.strptime(timeTaken, '%Y-%m-%d %H:%M:%S').timetuple()))
        closestId=findClosestPOI(lat,lon,pois)
        if closestId != -1:
            if owner not in users:
                users[owner]=[]
            users[owner].append((closestId,timeStamp))
    except:
        continue

for userId in users:
    users[userId].sort(key=lambda x:x[1])
toRemove=[]
for userId in users:
    if (users[userId][-1][1]-users[userId][0][1]>864000): #Remove users who stay longer than 10 days
        toRemove.append(userId)
for userId in toRemove:
    users.pop(userId)

toRemove=[]
for userId in users:
    firstId=users[userId][0][0]
    onePOI=True
    for pair in users[userId]:
        currId=pair[0]
        if currId!=firstId:
            onePOI=False
            break
    if onePOI==True:
        toRemove.append(userId)
for userId in toRemove:
    users.pop(userId)

stayTimes={}
with open('Itineraries.txt','w') as f:
    for userId in users:
        f.write(str(userId)+' ')
        oldId=users[userId][0][0]
        newId=0
        timeSpent=0
        oldTime=users[userId][0][1]
        numOfPhotos=0
        for poiPair in users[userId]:
            newId=poiPair[0]
            if newId!=oldId:
                if oldId not in stayTimes:
                    stayTimes[oldId]=[]
                if timeSpent==0:
                    timeSpent=10
                if timeSpent>86400:
                    timeSpent=10
                stayTimes[oldId].append(timeSpent)
                f.write(str(oldId)+';'+str(timeSpent)+';'+str(numOfPhotos)+' ')
                numOfPhotos=0
                oldTime=poiPair[1]
            else:
                numOfPhotos+=1
                timeSpent=poiPair[1]-oldTime
            oldId=newId
        if oldId not in stayTimes:
            stayTimes[oldId]=[]
        if timeSpent==0:
            timeSpent=10
        if timeSpent>86400:
            timeSpent=10
        stayTimes[oldId].append(timeSpent)
        f.write(str(oldId)+';'+str(timeSpent)+';'+str(numOfPhotos)+' ')
        f.write('\n')

with open('POIs_StayTime.txt','w') as f:
    for poiId in stayTimes:
        stayTimes[poiId]=np.mean(stayTimes[poiId])
        f.write(str(poiId)+' '+str(stayTimes[poiId])+'\n')

texts=list(rawTexts.values())

stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

procTexts=[]
for txt in texts:
    procTexts.append(preprocess(txt))

dictionary = gensim.corpora.Dictionary(procTexts)
dictionary.filter_extremes(no_below=15, no_above=0.9, keep_n=100000)

# count = 0
# for k, v in dictionary.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break

bow_corpus = [dictionary.doc2bow(doc) for doc in procTexts]
tfidf = models.TfidfModel(bow_corpus)

A=np.zeros((len(dictionary),len(bow_corpus)))
for txtInd,cor in enumerate(bow_corpus):
    for pair in tfidf[cor]:
        A[pair[0],txtInd]=pair[1]

u,s,vh = np.linalg.svd(A,full_matrices=False)
wikis=np.matmul(np.diag(s),vh)
wikis=wikis.T[:,:10]
count=1

with open('POIs_Vectors.txt','w') as f:
    for wiki in wikis:
        f.write(str(count)+' ')
        for num in wiki:
            f.write(str(num)+' ')
        f.write('\n')
        count+=1

with open('Users_Profiles.txt','w') as fw:
    with open('Itineraries.txt','r') as fr:
        spamreader=csv.reader(fr,delimiter=' ')
        for row in spamreader:
            vec=np.zeros(10)
            timeSpent=0
            for visit in row[1:-1]:
                print(visit)
                poiId,poiTime,numOfPhotos=map(int,visit.split(';'))
                vec=np.add(vec,[poiTime*x for x in wikis[poiId-1,:]])
                timeSpent+=poiTime
            vec=[x/timeSpent for x in vec]
            for num in vec:
                fw.write(str(num)+' ')
            fw.write('\n')
