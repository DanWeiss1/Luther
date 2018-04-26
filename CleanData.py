import re
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List

pd.options.display.max_rows = 999

df = pd.read_csv('Luther/sfbikes.csv')  # type: object

# Clean up price, so that it is numeric, fix outliers
df['price'] = df['price'].str.replace('$', '').astype(int)
df[df['price'] > 10000].head()
# drop obs with prices over 10000 bc not real (person wants bikes isn't selling them)
df = df[df['price'] <= 10000]

# obs with price <10 do not appear to be bikes or have accurate price
df = df[df['price'] >= 10]

# looks like a several obs below 40 are not bikes
dropTitles = ["Thomas Wooden Train Set!", "messenger  bag",
              "BOOMBOTIX REX BLUETOOTH SPEAKERS", "Misc: Bags Klickfix, Bell Dawn Patrol",
              "Sigma BC509 Bike Computer - BRAND NEW!", '2x bike stand',
              "Razor A5 Lux Scooter", "Brand New Diamondback Cycling jersey sz M",
              "Schwinn Bike Pump", "Bicycle Tires", '5/8" galvanized anchor chain',
              'Bicycle cardboard shipping box with divider', 'Ultegra Crankset 175mm FC6500',
              'Bike Stand for Two (2) Bikes', 'storage cabinets, office chairs, dresser',
              'Bike Sign * MAN-CAVE * Monark * Old-Time * Retro', 'New Maxxis Road Racing Tires',
              'Cargo Bar w/ hardware for Bikes', 'Allycat Shadow II Bike Trailer',
              '4 cross tires vittoria xg pro 34mm,xm pro 32mm', 'TRAILER',
              'Shimano SH-M 020D Womens Bicycling Shoes size 38 (5)',
              'EXY Trickstarter Scooter- Good for Beginner- Good Condition! Purple',
              'Ronstan nautical pulley and Genoa car', 'Minoura Inter Rim Trainer great shape!',
              'Car Bicycle Foldup Rack']
df = df[~df['title'].isin(dropTitles)]

# drop observations with helmet, rack, lock, parts in the title and less than threshhold price 
df = df[(df['title'].str.upper().str.find("HELMET") == -1) | (df['price'] > 40)]
df = df[(df['title'].str.upper().str.find("RACK") == -1) | (df.price > 60)]
df = df[(df.title.str.upper().str.find("LOCK") == -1) | (df.price > 30)]
df = df[df.title != "free bike parts"]

# Remove line breaks and generic text from post

df['textCleaner'] = df['text'].str.replace('\n', ' ').str.replace(
    'QR Code Link to This Post', '').str.strip()

df['postLength'] = df['textCleaner'].str.len()
df['postLength'].value_counts()
# get number of images
pattern = re.compile(r'((?<=image 1 of) \d*)')
df['num_pics'] = df['numPics'].str.extract(pattern, expand=False).str.strip().astype(float)
df['num_pics'] = np.where(df['num_pics'].isna(), 0, df['num_pics'])

# drop obs with price of 9500 (person requesting trade)
df.drop(df[df['price'] == 9500].index, inplace=True)
# extract condition from attributes
pattern = re.compile(r'((?<=condition: )[\w ]*(?=\n))')
df['condition'] = df['attributes'].str.extract(pattern, expand=False)
df['condition'] = np.where(df['condition'].isnull(), 'n/a', df['condition'])
df['condition'].value_counts()

# get dummies for condition but group salave with fair
df['condition'] = np.where(df['condition'] == "salvage", "fair", df['condition'])
df = pd.get_dummies(df, columns=["condition"])

# extract manufacturer from attributes
pattern = re.compile(r'((?<=make / manufacturer: )[\w ]*(?=\n))')
df['manufacturer'] = df['attributes'].str.extract(pattern, expand=False)
df['manufacturer'].value_counts()

# extract size
pattern = re.compile(r'((?<=size / dimensions: )[\w ]*(?=\n))')
df['size'] = df['attributes'].str.extract(pattern, expand=False)

# looks like one size is rear deraiuller not a bike but a part. drop
df = df[df['size'] != 'rear derailleur']

# extract model
pattern = re.compile(r'((?<=model name / number: )[\w ]*(?=\n))')
df['model'] = df['attributes'].str.extract(pattern, expand=False)
df['model'].value_counts()

df['model_listed'] = np.where(df['model'].isna(), 0, 1)
df['size_listed'] = np.where(df['size'].isna(), 0, 1)
df['brand_listed'] = np.where(df['manufacturer'].isna(), 0, 1)

# extract location from url
df['area'] = df['url'].str.replace('https://sfbay.craigslist.org/', '').str.slice(0, 3)

df = pd.get_dummies(df, columns=['area'])

# try to extract derailluer components
df['stringUpper'] = df['textCleaner'].str.upper()

pro_components = ['ETAP', 'SLX', 'XTR', 'RED', 'EAGLE', 'GX', 'XX', 'DURA ACE', 'RECORD', 'ACERA']
med_components = ['ULTEGRA', '105', 'FORCE', 'CHORUS', 'POTENZA', 'DEORE', 'X0',
                  'X9']
entry_components = ['TIAGRA', 'SORA', 'ALIVIO', 'ACERA', 'ALTUS', 'TOURNEY',
                    'X3', 'X4', 'X5', 'X7']

df['pro'] = np.where(df['stringUpper'].str.contains('|'.join(pro_components)), 1, 0)
df['pro'].value_counts()
df['med'] = np.where(df['stringUpper'].str.contains('|'.join(med_components)), 1, 0)
df[df['med'] == 1].price.describe()
df['entry'] = np.where(df['stringUpper'].str.contains('|'.join(entry_components)), 1, 0)
df[df['entry'] == 1].price.describe()
pd.crosstab(df['pro'], df['med'])

# create mutually exclusive categories Pro > Med > Entry > None
df['med'] = np.where(df['pro'] == 1, 0, df['med'])
df['entry'] = np.where((df['pro'] == 1) | (df['med'] == 1), 0, df['entry'])
df['no_component'] = np.where(df['pro'] + df['med'] + df['entry'] == 0, 1, 0)

# extract e bike or not
df['ebike'] = np.where(df['title'].str.upper().str.find('ELECTRIC') == -1, 0, 1)
df.ebike.value_counts()
df[df['ebike'] == 1]['price'].describe()
# extract kids vs adults
df['kids'] = np.where(df['textCleaner'].str.upper().str.find('KID') == -1, 0, 1)
df['kids'] = np.where(df['textCleaner'].str.upper().str.find('GIRL') == -1, df['kids'], 1)

df[df['kids'] == 1]['price'].describe()
df.price.describe()

df['logPrice'] = np.log10(df.price)
df.logPrice.hist()
df.price.describe()
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()
pd.to_pickle(df, 'Luther/bikedataset.pkl')
