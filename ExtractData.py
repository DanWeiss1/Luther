import copy
import math
import random
import re
import requests
import time

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy

# first get the total number of bikes for sale
p = '0'
url = requests.get('https://sfbay.craigslist.org/search/bik?s={0}&sort=date'.format(p))
soup = BeautifulSoup(url.text, 'lxml')
totCount = int(soup.find('span', class_="totalcount").text)  # type: int

# then set up loops to loop through each page of bikes and store all links in a list
max_page = math.ceil(totCount / 120) * 120
bike_link = re.compile(r"https\:\/\/sfbay\.craigslist\.org\/.*\/bik\/d.*")
links = []
for i in range(0, max_page, 120):
    time.sleep(random.uniform(0, 4))
    url = requests.get('https://sfbay.craigslist.org/search/bik?s=' + str(i) + '&sort=date')
    soup = BeautifulSoup(url.text, 'lxml')
    for link in soup.find_all('a'):
        if re.match(bike_link, link['href']) is not None:
            links.append(link['href'])

# de dup list
linksDeDup = list(set(links))
# get url, text, price, images, condition
# put into dictionary for each link and append to list of dicts
bikes = []
curDict = {}

for i, link in enumerate(linksDeDup):
    print("Starting bike %s" % i)
    time.sleep(random.uniform(0, 4))
    soup = BeautifulSoup(requests.get(link).text, 'lxml')
    # If no price skip to next link bc I need a price for the data to matter
    try:
        curDict['price'] = soup.find('span', class_="price").text
    except:
        continue
    curDict['url'] = link
    curDict['text'] = soup.find(id='postingbody').text
    curDict['title'] = soup.find(id='titletextonly').text
    try:
        curDict['location'] = soup.find('small').text
    except:
        curDict['location'] = None
    try:
        curDict['numPics'] = soup.find('span', class_='slider-info').text
    except:
        curDict['numPics'] = None
    # attributes
    try:
        curDict['attributes'] = soup.find('p', class_='attrgroup').text
    except:
        curDict['attributes'] = None

    di = copy.deepcopy(curDict)

    bikes.append(di)
    print('finished %s bike' % i)

df = pd.DataFrame(bikes)
df.to_csv('Luther/sfbikes.csv')
