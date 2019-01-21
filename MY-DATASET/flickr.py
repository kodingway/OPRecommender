import flickrapi
import wikipedia
import time
import datetime

centerName='Omonoia, Athens'
page=wikipedia.page(centerName)
centerLat=float(page.coordinates[0])
centerLon=float(page.coordinates[1])

api_key='8d1a7267a8e07b28c8997b8bf7ccabe8'
api_secret='66a3513acdb609b2'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
photos = flickr.photos.search(min_taken_date='2010-01-01',accuracy=16,lat=centerLat,lon=centerLon,radius=10)
numOfPages = photos['photos']['pages']

#GETTING ALL PHOTOS
photoDict=[]

for page in range(1,numOfPages+1):
    print(page)
    try:
        photos = flickr.photos.search(min_taken_date='2010-01-01', page=page,accuracy=16,lat=centerLat,lon=centerLon,radius=10)
        for photo in photos['photos']['photo']:
                photoDict.append(photo)
    except:
        continue

#CREATING SETS OF PHOTOS PER USER
usersDict={}
for photo in photoDict:
    usersDict[photo['owner']]=[]

for photo in photoDict:
    usersDict[photo['owner']].append(photo['id'])

#REMOVING USERS WITH LESS THAN 2 PHOTOS
toRemove=[]
for user in usersDict:
    if(len(usersDict[user])<2):
        toRemove.append(user)

for user in toRemove:
    usersDict.pop(user)

usersRoute={}
wikiPages=[]
for ind,user in enumerate(usersDict):
    print(ind)
    usersRoute[user]=[]
    toRemove=[]
    for photo in usersDict[user]:
        info = flickr.photos.getInfo(api_key=api_key,photo_id=photo)
        lat=float(info['photo']['location']['latitude'])
        lon=float(info['photo']['location']['longitude'])
        timeTaken=info['photo']['dates']['taken']
        timeStamp=int(time.mktime(datetime.datetime.strptime(timeTaken, '%Y-%m-%d %H:%M:%S').timetuple()))
        closest=wikipedia.geosearch(lat,lon,results=1,radius=200)
        if closest!=[]:
            for title in closest:
                page=wikipedia.page(title)
                coor=page.coordinates
                if ((title,coor) not in wikiPages):
                    wikiPages.append((title,coor))
            usersRoute[user].append({'wikiName':title,'time':timeStamp})