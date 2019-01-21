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

centerName='Omonoia, Athens'
page=wikipedia.page(centerName)
centerLat=float(page.coordinates[0])
centerLon=float(page.coordinates[1])

api_key='8d1a7267a8e07b28c8997b8bf7ccabe8'
api_secret='66a3513acdb609b2'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
photos = flickr.photos.search(min_taken_date='2017-01-01',accuracy=16,lat=centerLat,lon=centerLon,radius=10,in_gallery=True)
numOfPages = photos['photos']['pages']

#GETTING ALL PHOTOS
photoDict=[]

for page in range(1,numOfPages+1):
    print(page)
    try:
        photos = flickr.photos.search(min_taken_date='2017-01-01', page=page,accuracy=16,lat=centerLat,lon=centerLon,radius=10,in_gallery=True)
        for photo in photos['photos']['photo']:
                photoDict.append(photo)
    except:
        continue

pois={}
rawTexts={}
poiTitleToId={}
users={}

myId=1
for ind,photo in enumerate(photoDict):
    print(ind)
    if ind==20:
        break
    try:
        info = flickr.photos.getInfo(api_key=api_key,photo_id=photo['id'])
        owner=info['photo']['owner']['nsid']
        lat=float(info['photo']['location']['latitude'])
        lon=float(info['photo']['location']['longitude'])
        timeTaken=info['photo']['dates']['taken']
        timeStamp=int(time.mktime(datetime.datetime.strptime(timeTaken, '%Y-%m-%d %H:%M:%S').timetuple()))
        closest=wikipedia.geosearch(lat,lon,results=1,radius=200)
        if closest!=[]:
            if closest[0] not in poiTitleToId:
                page=wikipedia.page(closest[0])
                try:
                    plat,plon=map(float,page.coordinates)
                except:
                    plat=lat
                    plon=lon
                poiTitleToId[closest[0]]=myId
                pois[myId]={'name':closest[0],'lat':plat,'lon':plon}
                rawTexts[myId]=page.content
                myId+=1
            poiId=poiTitleToId[closest[0]]
            if owner not in users:
                users[owner]=[]
            users[owner].append((poiId,timeStamp))
    except:
        continue

with open('dictionary.txt','w') as f:
    for title in poiTitleToId:
        f.write(title+'\t'+str(poiTitleToId[title])+'\t'+str(pois[poiTitleToId[title]]['lat'])+' '+str(pois[poiTitleToId[title]]['lon'])+'\n')

toRemove=[]
for userId in users:
    if len(users[userId])==1:   #Remove users with just one photo
        toRemove.append(userId)
for userId in toRemove:
    users.pop(userId)

for userId in users:
    users[userId].sort(key=lambda x:x[1])
toRemove=[]
for userId in users:
    if (users[userId][-1][1]-users[userId][0][1]>864000): #Remove users who stay longer than 10 days
        toRemove.append(userId)
for userId in toRemove:
    users.pop(userId)

with open('Distances.txt','w') as f:
    for poiI in pois:
        for poiJ in pois:
            latI=radians(pois[poiI]['lat'])
            lonI=radians(pois[poiI]['lon'])
            latJ=radians(pois[poiJ]['lat'])
            lonJ=radians(pois[poiJ]['lon'])
            dlon = lonJ - lonI
            dlat = latJ - latI
            a = sin(dlat/2)**2 + cos(latI)*cos(latJ)*sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            R=6373.0
            f.write(str(poiI)+' '+str(poiJ)+' '+str(R*c)+'\n')

stayTimes={}
for userId in users:
    oldId=users[userId][0][0]
    newId=0
    timeSpent=0
    oldTime=users[userId][0][1]
    for poiPair in users[userId]:
        newId=poiPair[0]
        if newId!=oldId:
            if oldId not in stayTimes:
                stayTimes[oldId]=[]
            if timeSpent==0:
                timeSpent=10
            stayTimes[oldId].append(timeSpent)
            oldTime=poiPair[1]
        else:
            timeSpent=poiPair[1]-oldTime
        oldId=newId

with open('POIs_StayTime.txt','w') as f:
    for poiId in stayTimes:
        stayTimes[poiId]=np.mean(stayTimes[poiId])
        f.write(str(poiId)+' '+str(stayTimes[poiId])+'\n')

texts=list(rawTexts.values())
nltk.download('wordnet')

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
#wikis=np.matmul(np.diag(s),vh)  #XREIAZETAI????
wikis=vh
wikis=wikis.T[:,:10]
count=1

with open('POIs_Vectors.txt','w') as f:
    for wiki in wikis:
        f.write(str(count)+' ')
        for num in wiki:
            f.write(str(num)+' ')
        f.write('\n')
        count+=1

with open('Users_Profiles.txt','w') as f:
    for userId in users:
        numPhotos=0
        vec=np.zeros(10)
        for pair in users[userId]:
            poiId=pair[0]
            vec=np.add(vec,wikis[poiId-1,:])
            numPhotos+=1
        vec=[x/numPhotos for x in vec]
        for num in vec:
            f.write(str(num)+' ')
        f.write('\n')