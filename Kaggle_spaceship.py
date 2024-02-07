from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from ydata_profiling import ProfileReport
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance, XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import scipy.stats as stats
import seaborn as sns
import optuna as opt
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import json



pd.set_option('display.max_columns', None)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


train_df = pd.read_csv('train_spaceship.csv')
test_df = pd.read_csv('test_spaceship.csv')
test_df_copy = test_df.copy()

# -2) General overview of the data: -------------------------------------------------------------------------------------------
train_profile = ProfileReport(train_df)
train_profile.to_file('profile_train')
test_profile = ProfileReport(test_df)
test_profile.to_file('profile_test')


# -1) Creating necessary new feature Ticket ---------------------------------------------------------------------------
train_df['Ticket'] = train_df['PassengerId'].apply(lambda x: str(x).split('_')[0])
test_df['Ticket'] = test_df['PassengerId'].apply(lambda x: str(x).split('_')[0])

# 0) Setting unknown HomePlanet value to the value assigned for a person with the same ticket number (if applies)
# surely not the ideal way to enumerate through df, but it works - optimize it later!

for i, (ticket, home_planet) in enumerate(zip(train_df.Ticket, train_df.HomePlanet)):
    try:
        check1 = train_df.loc[i - 1][14]
        check2 = train_df.loc[i + 1][14]
    except:
        pass
    try:
        if i > 0 and check1 == ticket and math.isnan(home_planet):
            train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Ticket'] == ticket),
                                              train_df.iloc[i - 1][1], train_df['HomePlanet'])
        elif i < (len(train_df.Ticket) - 1) and check2 == ticket and math.isnan(home_planet):
            try:
                train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Ticket'] == ticket),
                                                  train_df.iloc[i + 1][1], train_df['HomePlanet'])
            except:
                pass
    except:
        pass

for i, (ticket, home_planet) in enumerate(zip(test_df.Ticket, test_df.HomePlanet)):
    try:
        check1 = test_df.loc[i - 1][14]
        check2 = test_df.loc[i + 1][14]
    except:
        pass
    try:
        if i > 0 and check1 == ticket and math.isnan(home_planet):
            test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Ticket'] == ticket),
                                              test_df.iloc[i - 1][1], test_df['HomePlanet'])
        elif i < (len(test_df.Ticket) - 1) and check2 == ticket and math.isnan(home_planet):
            try:
                test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Ticket'] == ticket),
                                                  test_df.iloc[i + 1][1], test_df['HomePlanet'])
            except:
                pass
    except:
        pass


# 1) Filling NaNs  (HomePlanet)-----------------------------------------------------------------------------------------

# a) Filling Name with Unknown Unknown -------------------------------------------------------------------------------
train_df['Name'].fillna('Unknown Unknown', inplace=True)
test_df['Name'].fillna('Unknown Unknown', inplace=True)

# b) A bit of early, necessary feature engineering. Creating 2 new features Deck and Side ------------------------------
train_df['Cabin'].fillna('None/None/None', inplace=True)
train_df['Deck'] = train_df['Cabin'].apply(lambda x: str(x).split('/')[0])
train_df['Number'] = train_df['Cabin'].apply(lambda x: str(x).split('/')[1])
train_df['Number'] = np.where((train_df['Number'] == 'None'), '99999', train_df['Number'])
train_df['Number'] = train_df['Number'].apply(int)
train_df['Side'] = train_df['Cabin'].apply(lambda x: str(x).split('/')[2])
test_df['Cabin'].fillna('None/None/None', inplace=True)
test_df['Deck'] = test_df['Cabin'].apply(lambda x: str(x).split('/')[0])
test_df['Number'] = test_df['Cabin'].apply(lambda x: str(x).split('/')[1])
test_df['Number'] = np.where((test_df['Number'] == 'None'), '99999', test_df['Number'])
test_df['Number'] = test_df['Number'].apply(int)
test_df['Side'] = test_df['Cabin'].apply(lambda x: str(x).split('/')[2])

# c1) Number - part 1 --------------------------------------------------------------------------------------------------
# Logic: same ticket = same cabin number. At least for most of the time, which is enough for us.
for i, (ticket, number) in enumerate(zip(train_df.Ticket, train_df.Number)):
    try:
        check3 = train_df.loc[i - 1][14]
        check4 = train_df.loc[i + 1][14]
    except:
        pass
    try:
        if i > 0 and check3 == ticket and number == 99999:
            train_df['Number'] = np.where((train_df['Number'] == 99999) & (train_df['Ticket'] == ticket),
                                              train_df.iloc[i - 1][16], train_df['Number'])
        elif i < (len(train_df.Ticket) - 1) and check4 == ticket and number == 99999:
            try:
                train_df['Number'] = np.where((train_df['Number'] == 99999) & (train_df['Ticket'] == ticket),
                                                  train_df.iloc[i + 1][16], train_df['Number'])
            except:
                pass
    except:
        pass


for i, (ticket, number) in enumerate(zip(test_df.Ticket, test_df.Number)):
    try:
        check3 = test_df.loc[i - 1][13]
        check4 = test_df.loc[i + 1][13]
    except:
        pass
    try:
        if i > 0 and check3 == ticket and number == 99999:
            test_df['Number'] = np.where((test_df['Number'] == 99999) & (test_df['Ticket'] == ticket),
                                              test_df.iloc[i - 1][15], test_df['Number'])
        elif i < (len(train_df.Ticket) - 1) and check4 == ticket and number == 99999:
            try:
                test_df['Number'] = np.where((test_df['Number'] == 99999) & (test_df['Ticket'] == ticket),
                                                  test_df.iloc[i + 1][15], test_df['Number'])
            except:
                pass
    except:
        pass


# c2) Deck unique ---------------------------------------------------------------------------------------------------------
# On these decks people only from specific planet are present.
for i in ['A','B','C','T']:
    train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df.Deck == i), 'Europa',
                                  train_df['HomePlanet'])

train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df.Deck == 'G'), 'Earth',
                                  train_df['HomePlanet'])

for i in ['A','B','C','T']:
    test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df.Deck == i), 'Europa',
                                  test_df['HomePlanet'])

test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df.Deck == 'G'), 'Earth',
                                  test_df['HomePlanet'])


# d) Surname length --------------------------------------------------------------------------------------------------
# Filling HomePlanet as a Mars where len(Name[1]) in (3,4) - only Marsians have got that short surnames.
train_df['HomePlanet'] = np.where((train_df['Name'].apply(lambda x: len(str(x).split(' ')[1])) == 3) & (train_df['HomePlanet'].isna()),
                                  'Mars',train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['Name'].apply(lambda x: len(str(x).split(' ')[1])) == 4) & (train_df['HomePlanet'].isna()),
                                  'Mars',train_df['HomePlanet'])

test_df['HomePlanet'] = np.where((test_df['Name'].apply(lambda x: len(str(x).split(' ')[1])) == 3) & (test_df['HomePlanet'].isna()),
                                  'Mars',test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['Name'].apply(lambda x: len(str(x).split(' ')[1])) == 4) & (test_df['HomePlanet'].isna()),
                                  'Mars',test_df['HomePlanet'])


# e) Total Name length ----------------------------------------------------------------------------------------------------------------------
# Finding out that on Deck D there are only people from Europa and Mars. Interestingly both groups have some specific
# fullname lengths overally (same for train and test). So if total length is 7,8,9,10 - NaN is from Mars, if 13,14,16,17,18 - from Europa.
# The same logic applied to Deck F. 7,8 - Mars. 13,14,16,17,18 - Earth.

print(np.unique(train_df['Name'][train_df['HomePlanet'] == 'Earth'].apply(lambda x: len(str(x)))))
print(np.unique(train_df['Name'][train_df['HomePlanet'] == 'Europa'].apply(lambda x: len(str(x)))))
print(np.unique(train_df['Name'][train_df['HomePlanet'] == 'Mars'].apply(lambda x: len(str(x)))))
print(train_df.groupby(['Deck', 'HomePlanet']).size())
#
print(np.unique(test_df['Name'][test_df['HomePlanet'] == 'Earth'].apply(lambda x: len(str(x)))))
print(np.unique(test_df['Name'][test_df['HomePlanet'] == 'Europa'].apply(lambda x: len(str(x)))))
print(np.unique(test_df['Name'][test_df['HomePlanet'] == 'Mars'].apply(lambda x: len(str(x)))))
print(test_df.groupby(['Deck', 'HomePlanet']).size())
print(test_df[test_df['HomePlanet'].isna()])


# There is no NaNs left on the deck D with 7,8,9 or 10 name total, but there are a few NaNs or unknown(None) deck. Later
# applying logic described above.
train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Deck'] == 'None')
                                  & (train_df['Name'].apply(lambda x: len(str(x))).isin([7, 8, 9, 10])), 'Mars',
                                  train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Deck'] == 'D')
                                  & (train_df['Name'].apply(lambda x: len(str(x))).isin([13,14,16,17,18])), 'Europa',
                                  train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Deck'] == 'None')
                                  & (train_df['Name'].apply(lambda x: len(str(x))).isin([13,14,16,17,18])), 'Europa',
                                  train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Deck'] == 'F')
                                  & (train_df['Name'].apply(lambda x: len(str(x))).isin([13,14,16,17,18])), 'Europa',
                                  train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) & (train_df['Deck'] == 'F')
                                  & (train_df['Name'].apply(lambda x: len(str(x))).isin([7, 8, 9, 10])), 'Mars',
                                  train_df['HomePlanet'])

test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Deck'] == 'None')
                                  & (test_df['Name'].apply(lambda x: len(str(x))).isin([7, 8, 9, 10])), 'Mars',
                                  test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Deck'] == 'D')
                                  & (test_df['Name'].apply(lambda x: len(str(x))).isin([13,14,16,17,18])), 'Europa',
                                  test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Deck'] == 'None')
                                  & (test_df['Name'].apply(lambda x: len(str(x))).isin([13,14,16,17,18])), 'Europa',
                                  test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Deck'] == 'F')
                                  & (test_df['Name'].apply(lambda x: len(str(x))).isin([13,14,16,17,18])), 'Europa',
                                  test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) & (test_df['Deck'] == 'F')
                                  & (test_df['Name'].apply(lambda x: len(str(x))).isin([7, 8, 9, 10])), 'Mars',
                                  test_df['HomePlanet'])

# Names that don't sound Earthish with length constraints taken into account.
test_df['HomePlanet'] = np.where((test_df['Name'] == 'Zedares Maltorted'), 'Europa', test_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['Name'] == 'Rassias Freednal'), 'Europa', train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['Name'] == 'Arkaban Tertchty'), 'Europa', train_df['HomePlanet'])


# f) Specific surname suffixes -----------------------------------------------------------------------------------------
# People from Mars and Europe have set of unique surname suffixes (last 3 letters) - checked also on test set for sanity,
# but analyzed independently. So you can say that it might have created a very small data leakage.

# Train data ----------------------------------------------------------------------------------------------------------
m = np.unique(train_df['Name'][train_df['HomePlanet'] == 'Mars'].apply(lambda x: str(x).split(' ')[1][-3:]))
m = [x.lower() for x in m]
e = np.unique(train_df['Name'][train_df['HomePlanet'] == 'Earth'].apply(lambda x: str(x).split(' ')[1][-3:]))
e = [x.lower() for x in e]
u = np.unique(train_df['Name'][train_df['HomePlanet'] == 'Europa'].apply(lambda x: str(x).split(' ')[1][-3:]))
u = [x.lower() for x in u]

# g) checking suffixes from 1 planet against another 2 planets. Finding out unique ones.
unique_E = []
mu = np.hstack((m,u))
for i in e:
    if i not in mu:
        unique_E.append(i)

unique_U = []
me = np.hstack((m,e))
for i in u:
    if i not in me:
        unique_U.append(i)

unique_M = []
ue = np.hstack((u,e))
for i in m:
    if i not in ue:
        unique_M.append(i)


# h) Filling NaNs where unique suffix is spotted
train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) &
                                  (train_df['Name'].apply(lambda x: str(x).split(' ')[1][-3:].lower()).isin(unique_M)), 'Mars',
                                  train_df['HomePlanet'])

train_df['HomePlanet'] = np.where((train_df['HomePlanet'].isna()) &
                                  (train_df['Name'].apply(lambda x: str(x).split(' ')[1][-3:].lower()).isin(unique_E)), 'Earth',
                                  train_df['HomePlanet'])

# Test data ---------------------------------------------------------------------------------------------------------------
m = np.unique(test_df['Name'][test_df['HomePlanet'] == 'Mars'].apply(lambda x: str(x).split(' ')[1][-3:]))
m = [x.lower() for x in m]
e= np.unique(test_df['Name'][test_df['HomePlanet'] == 'Earth'].apply(lambda x: str(x).split(' ')[1][-3:]))
e = [x.lower() for x in e]
u = np.unique(test_df['Name'][test_df['HomePlanet'] == 'Europa'].apply(lambda x: str(x).split(' ')[1][-3:]))
u = [x.lower() for x in u]


unique_E_test = []
mu = np.hstack((m,u))
for i in e:
    if i not in mu:
        unique_E_test.append(i)

unique_U_test = []
me = np.hstack((m,e))
for i in u:
    if i not in me:
        unique_U_test.append(i)

unique_M_test= []
ue = np.hstack((u,e))
for i in m:
    if i not in ue:
        unique_M_test.append(i)

test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) &
                                  (test_df['Name'].apply(lambda x: str(x).split(' ')[1][-3:].lower()).isin(unique_M_test)), 'Mars',
                                  test_df['HomePlanet'])

test_df['HomePlanet'] = np.where((test_df['HomePlanet'].isna()) &
                                  (test_df['Name'].apply(lambda x: str(x).split(' ')[1][-3:].lower()).isin(unique_E_test)), 'Earth',
                                  test_df['HomePlanet'])


# i) Right now I have no more ideas in the bag. Some kind of stemming or lemmatization might be useful, but since the NaNs number
# in HomePlanet is so low, I will do it by eye test, and later will probably polish it.

# Name eye test and Deck constraints
train_names = ['Adrie Rodger', 'Sharie Rodricker', 'Stevey Reilline', 'Morrie Moongton', 'Idarry Norrison']
train_names_M = ['Sealfs Sutty']
for name in train_names:
    train_df['HomePlanet'] = np.where((train_df['Name'] == name), 'Earth', train_df['HomePlanet'])

for name in train_names_M:
    train_df['HomePlanet'] = np.where((train_df['Name'] == name), 'Mars', train_df['HomePlanet'])

test_names = ['Kell Holton', 'Racey Daughessey', 'Rayle Bradlerson', 'Joela Bonder', 'Andine Warrishales']
test_names_M = ['Flams Blane', 'Miten Emone']
for name in test_names:
    test_df['HomePlanet'] = np.where((test_df['Name'] == name), 'Earth', test_df['HomePlanet'])

for name in test_names_M:
    test_df['HomePlanet'] = np.where((test_df['Name'] == name), 'Mars', test_df['HomePlanet'])

# VRDeck. 2500+ Europa dominates
# print(train_df.groupby(['HomePlanet', pd.cut(train_df['VRDeck'], np.arange(0, 5000, 500))]).size().unstack(0))
# train_df.groupby(['HomePlanet', pd.cut(train_df['VRDeck'],
#                                        np.arange(0, 5000, 500))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
# plt.show()

train_df['HomePlanet'] = np.where((train_df['Name'] == 'Kocha Cluitty'), 'Europa', train_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['Name'] == 'Gieba Timanable'), 'Europa', test_df['HomePlanet'])

# RoomService and Deck constraints. Typically higher for young people from Mars.
test_df['HomePlanet'] = np.where((test_df['RoomService'] == 2195.0) & (test_df['Ticket'] == '6499'), 'Mars', test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['Name'] == 'Quants Burle'), 'Mars', test_df['HomePlanet'])

# FoodCourt and Deck constraints
train_df['HomePlanet'] = np.where((train_df['Name'] == 'Nuscham Stingive'), 'Europa', train_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['Name'] == 'Neutrix Watuald'), 'Europa', test_df['HomePlanet'])

# VRDeck and Age name style(optional)
test_df['HomePlanet'] = np.where((test_df['Name'] == 'Andan Fryan'), 'Earth', test_df['HomePlanet'])
test_df['HomePlanet'] = np.where((test_df['Ticket'] == '8435'), 'Earth', test_df['HomePlanet'])

# PassengerId, because of group ticket
test_df['HomePlanet'] = np.where((test_df['Ticket'] == '6559') & (test_df['RoomService'] == 252.0), 'Earth', test_df['HomePlanet'])

# Destination and Age
train_df['HomePlanet'] = np.where((train_df['Name'] == 'Teron Sageng'), 'Europa', train_df['HomePlanet'])

# Maybe name style, age and deck mean
train_df['HomePlanet'] = np.where((train_df['Name'] == 'Eilan Solivers' ), 'Earth', train_df['HomePlanet'])

# All together: RoomService, ShoppinMall, Destination
train_df['HomePlanet'] = np.where((train_df['Ticket'] == '2443') & (train_df['FoodCourt'] == 421.0), 'Mars', train_df['HomePlanet'])

# Purely on feature median
train_df['HomePlanet'] = np.where((train_df['Ticket'] == '3331') & (train_df['FoodCourt'] == 4.0), 'Earth', train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['Ticket'] == '4840') & (train_df['Age'] == 36.0), 'Earth', train_df['HomePlanet'])
train_df['HomePlanet'] = np.where((train_df['Ticket'] == '6108') & (train_df['Age'] == 13.0), 'Earth', train_df['HomePlanet'])

# No more NaNs in HomePlanet, at last!!!!!!!!!!!


# 2) Filling NaNs --(All expenses like Spa etc)-----------------------------------------------------------------------------------------

# a) Creating temp, new feature TotalExpenses. Fillna(0) to avoid getting total value = nan when in some column is nan.
train_df['TotalExpenses'] = (train_df['RoomService'].fillna(0) + train_df['FoodCourt'].fillna(0) +
                             train_df['ShoppingMall'].fillna(0) + train_df['Spa'].fillna(0) + train_df['VRDeck'].fillna(0))
test_df['TotalExpenses'] = (test_df['RoomService'].fillna(0) + test_df['FoodCourt'].fillna(0) +
                             test_df['ShoppingMall'].fillna(0) + test_df['Spa'].fillna(0) + test_df['VRDeck'].fillna(0))

# b) Filling nans with zero when passenger is in CryoSleep
for feature in ['RoomService', 'FoodCourt', 'ShoppingMall','Spa', 'VRDeck']:
    train_df[feature] = np.where((train_df['CryoSleep'] == True) & (train_df[feature].isna()), 0.0, train_df[feature])

for feature in ['RoomService', 'FoodCourt', 'ShoppingMall','Spa', 'VRDeck']:
    test_df[feature] = np.where((test_df['CryoSleep'] == True) & (test_df[feature].isna()), 0.0, test_df[feature])

# c) 50/8693(test: 0/4277) passengers have paid only for RoomService and 3653/8693(test: 1804/4277)
# passengers have 0 in TotalExpenses, so filling NaN with 0
train_df['RoomService'] = np.where((train_df['RoomService'] != 0) & (train_df['FoodCourt'] == 0) & (train_df['ShoppingMall'] == 0)
               & (train_df['Spa'] == 0) & (train_df['VRDeck'] == 0), 0.0, train_df['RoomService'])
test_df['RoomService'] = np.where((test_df['RoomService'] != 0) & (test_df['FoodCourt'] == 0) & (test_df['ShoppingMall'] == 0)
               & (test_df['Spa'] == 0) & (test_df['VRDeck'] == 0), 0.0, test_df['RoomService'])

# d) 37/8693(test: 23/4277) passengers have paid only for FoodCourt and 3653/8693(test: 1804/4277) passengers have 0 in TotalExpenses,
# so filling NaN with 0
train_df['FoodCourt'] = np.where((train_df['FoodCourt'] != 0) & (train_df['RoomService'] == 0) & (train_df['ShoppingMall'] == 0)
               & (train_df['Spa'] == 0) & (train_df['VRDeck'] == 0), 0.0, train_df['FoodCourt'])
test_df['FoodCourt'] = np.where((test_df['FoodCourt'] != 0) & (test_df['RoomService'] == 0) & (test_df['ShoppingMall'] == 0)
               & (test_df['Spa'] == 0) & (test_df['VRDeck'] == 0), 0.0, test_df['FoodCourt'])

# e) 49/8693(test: 21/4277) passengers have paid only for ShoppingMall and 3653/8693(test: 1804/4277) passengers have 0 in TotalExpenses,
# so filling NaN with 0
train_df['ShoppingMall'] = np.where((train_df['ShoppingMall'] != 0) & (train_df['RoomService'] == 0) & (train_df['RoomService'] == 0)
               & (train_df['Spa'] == 0) & (train_df['VRDeck'] == 0), 0.0, train_df['ShoppingMall'])
test_df['ShoppingMall'] = np.where((test_df['ShoppingMall'] != 0) & (test_df['RoomService'] == 0) & (test_df['RoomService'] == 0)
               & (test_df['Spa'] == 0) & (test_df['VRDeck'] == 0), 0.0, test_df['ShoppingMall'])

# f) 0 cases like that in Spa column, (test: 15/4277) and (test: 1804/4277) passengers have 0 in TotalExpenses,
# # so filling NaN with 0
test_df['Spa'] = np.where((test_df['Spa'] != 0) & (test_df['RoomService'] == 0) & (test_df['RoomService'] == 0)
               & (test_df['ShoppingMall'] == 0) & (test_df['VRDeck'] == 0), 0.0, test_df['Spa'])

# g) 143/8693(test: 10/4277) passengers have paid only for VRDeck and 3653/8693(test: 1804/4277) passengers have 0 in TotalExpenses, the number of cases
# is 3 times larger but still not significant, so filling NaN with 0, so filling NaN with 0
train_df['VRDeck'] = np.where((train_df['VRDeck'] != 0) & (train_df['RoomService'] == 0) & (train_df['ShoppingMall'] == 0)
               & (train_df['FoodCourt'] == 0) & (train_df['Spa'] == 0), 0.0, train_df['VRDeck'])
test_df['VRDeck'] = np.where((test_df['VRDeck'] != 0) & (test_df['RoomService'] == 0) & (test_df['ShoppingMall'] == 0)
               & (test_df['FoodCourt'] == 0) & (test_df['Spa'] == 0), 0.0, test_df['VRDeck'])

# COMMENT: later try to fill all these expenses with mean to check if the score would be better. Maybe single NaN in 1 column
# suggest specific non zero expense not just a random error

# h) RoomService, FoodCourt, ShoppingMall, Spa,VRDeck.
# Filling NaNs with mean of nonzero values. The idea behind is that, if a passenger paid for other services,
# there is higher probability that he paid also for roomservice.
for feature in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    # Creating copy of train_df. Calculating mean for nonzero values grouped by HomePlanet and Age(bins)
    for df in [train_df, test_df]:
        temp_train_df = df.copy()
        temp_train_df.drop(temp_train_df[temp_train_df[feature] == 0].index, inplace=True)
        temp_train_df[feature] = temp_train_df[feature].fillna(temp_train_df.groupby(['HomePlanet', pd.cut(temp_train_df['Age'],
                                    np.arange(0, 100, 5))])[feature].transform('mean'))


        # We need to find matching something, let it be a new index set on PassengerId (indexes of two data frames are
        # because of the drop RoomService == 0
        df.set_index('PassengerId', inplace=True)
        df.update(temp_train_df.set_index('PassengerId'))
        # Recovering initial structure
        df.reset_index(inplace=True)

# i) RoomService, FoodCourt, ShoppingMall, Spa,VRDeck.by % of TotalExpenses
# Finding out that ShoppingMall constitutes on average 31% of TotalExpenses and so on, not taking into account 0 cells:
train_df.drop(train_df[train_df['ShoppingMall'] == 0].index, inplace=True)
train_df.drop(train_df[train_df['TotalExpenses'] == 0].index, inplace=True)
print((train_df['ShoppingMall']/train_df['TotalExpenses']).mean())

# QUESTION - what is worse: joining test and train data for calculating means and percentages or transorming train and test
# data using their own numbers?

train_df['ShoppingMall'] = np.where((train_df['ShoppingMall'].isna()), ((train_df['TotalExpenses'] * 0.31)/0.69),
                                     train_df['ShoppingMall'])
train_df['Spa'] = np.where((train_df['Spa'].isna()) & (train_df['TotalExpenses'] != 0), ((train_df['TotalExpenses'] * 0.55)/0.45),
                                     train_df['Spa'])
train_df['VRDeck'] = np.where((train_df['VRDeck'].isna()) & (train_df['TotalExpenses'] != 0), ((train_df['TotalExpenses'] * 0.36)/0.64),
                                     train_df['VRDeck'])

test_df['VRDeck'] = np.where((test_df['VRDeck'].isna()) & (test_df['TotalExpenses'] != 0), ((test_df['TotalExpenses'] * 0.32)/0.68),
                                     test_df['VRDeck'])
test_df['ShoppingMall'] = np.where((test_df['ShoppingMall'].isna()), ((test_df['TotalExpenses'] * 0.55)/0.45),
                                     test_df['ShoppingMall'])
test_df['FoodCourt'] = np.where((test_df['FoodCourt'].isna()) & (test_df['TotalExpenses'] != 0), ((test_df['TotalExpenses'] * 0.48)/0.52),
                                     test_df['FoodCourt'])
test_df['Spa'] = np.where((test_df['Spa'].isna()) & (test_df['TotalExpenses'] != 0), ((test_df['TotalExpenses'] * 0.32)/0.68),
                                     test_df['Spa'])

# Rest filled with 0
for feature in ['FoodCourt', 'Spa', 'VRDeck']:
    train_df[feature].fillna(0, inplace=True)

# j) Updating TotalExpenses
train_df['TotalExpenses'] = (train_df['RoomService'].fillna(0) + train_df['FoodCourt'].fillna(0) +
                             train_df['ShoppingMall'].fillna(0) + train_df['Spa'].fillna(0) + train_df['VRDeck'].fillna(0))
test_df['TotalExpenses'] = (test_df['RoomService'].fillna(0) + test_df['FoodCourt'].fillna(0) +
                             test_df['ShoppingMall'].fillna(0) + test_df['Spa'].fillna(0) + test_df['VRDeck'].fillna(0))

# 3) Filling nan VIP --------------------------------------------------------------------------------------------------
# No VIPs from Earth at all, and no VIPs who hasn't paid a single dime.
train_df['VIP'] = np.where((train_df['VIP'].isna()) & (train_df['HomePlanet'] == 'Earth'), False, train_df['VIP'])
train_df['VIP'] = np.where((train_df['VIP'].isna()) & (train_df['TotalExpenses'] == 0), False, train_df['VIP'])
test_df['VIP'] = np.where((test_df['VIP'].isna()) & (test_df['HomePlanet'] == 'Earth'), False, test_df['VIP'])
test_df['VIP'] = np.where((test_df['VIP'].isna()) & (test_df['TotalExpenses'] == 0), False, test_df['VIP'])

# Median TotalExpenses for Europa and Mars similar for VIPs and nonVIPs, so I set threshold for roughly 3x median.
train_df['VIP'] = np.where((train_df['VIP'].isna()) & (train_df['TotalExpenses'] > 16000) &
                           ( train_df['HomePlanet'] == 'Europa'), True, train_df['VIP'])
train_df['VIP'] = np.where((train_df['VIP'].isna()) & (train_df['TotalExpenses'] > 6000) &
                           ( train_df['HomePlanet'] == 'Mars'), True, train_df['VIP'])
test_df['VIP'] = np.where((test_df['VIP'].isna()) & (test_df['TotalExpenses'] > 18000) &
                           ( test_df['HomePlanet'] == 'Europa'), True, test_df['VIP'])

# Rest filled with most frequent = False
train_df['VIP'].fillna(False, inplace=True)
test_df['VIP'].fillna(False, inplace=True)

# 4) Filling nan Age -------------------------------------------------------------------------------------------------
# with Mean for a planet
train_df['Age'].fillna(train_df.groupby(['HomePlanet'])['Age'].transform('mean'), inplace=True)
test_df['Age'].fillna(test_df.groupby(['HomePlanet'])['Age'].transform('mean'), inplace=True)

# 5) Filling nan CryoSleep --------------------------------------------------------------------------------------------
train_df['CryoSleep'] = np.where((train_df['CryoSleep'].isna()) & (train_df['TotalExpenses'] > 0),False, train_df['CryoSleep'] )
train_df['CryoSleep'] = np.where((train_df['CryoSleep'].isna()) & (train_df['TotalExpenses'] == 0),True, train_df['CryoSleep'] )
test_df['CryoSleep'] = np.where((test_df['CryoSleep'].isna()) & (test_df['TotalExpenses'] > 0),False, test_df['CryoSleep'] )
test_df['CryoSleep'] = np.where((test_df['CryoSleep'].isna()) & (test_df['TotalExpenses'] == 0),True, test_df['CryoSleep'] )

# 6) Filling Destination ----------------------------------------------------------------------------------------------
train_df['Destination'].fillna('TRAPPIST-1e', inplace=True)
test_df['Destination'].fillna('TRAPPIST-1e', inplace=True)

# 7) Replacing None in Deck, Side with most frequent per HomePlanet and Destination -----------------------------------
train_df['Deck'] = np.where((train_df['Deck'] == 'None'), np.nan, train_df['Deck'])
train_df['Deck'] = train_df.groupby(['HomePlanet', 'Destination'])['Deck'].transform(lambda x: x.fillna(x.mode().iloc[0]))
train_df['Side'] = np.where((train_df['Side'] == 'None'), np.nan, train_df['Side'])
train_df['Side'] = train_df.groupby(['HomePlanet', 'Destination'])['Side'].transform(lambda x: x.fillna(x.mode().iloc[0]))

test_df['Deck'] = np.where((test_df['Deck'] == 'None'), np.nan, test_df['Deck'])
test_df['Deck'] = test_df.groupby(['HomePlanet', 'Destination'])['Deck'].transform(lambda x: x.fillna(x.mode().iloc[0]))
test_df['Side'] = np.where((test_df['Side'] == 'None'), np.nan, test_df['Side'])
test_df['Side'] = test_df.groupby(['HomePlanet', 'Destination'])['Side'].transform(lambda x: x.fillna(x.mode().iloc[0]))

"""Mode is not compatible with fillna in the same way as mean & median.
Mean & median returns both returns a series. But mode returns a dataframe.
To use mode with fillna we need make a little change. We need to locate the fist data using iloc.
 a = df.mode; print(type(a) -> <class ‘pandas.core.frame.DataFrame’>"""

# 8) Changing columns type from float to int, to avoid model bias on inputed decimals
for feature in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpenses']:
    train_df[feature] = train_df[feature].apply(int)

for feature in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpenses']:
    test_df[feature] = test_df[feature].apply(int)

# 8,5) Number - Part 2. ----------------------------------------------------------------------------------------------
# Filling with median for the corresponding Deck
train_df['Number'] = np.where((train_df['Number'] == 99999),
                              train_df.groupby(['Deck'])['Number'].transform(lambda x: x.median()),train_df['Number'])
test_df['Number'] = np.where((test_df['Number'] == 99999),
                              test_df.groupby(['Deck'])['Number'].transform(lambda x: x.median()),test_df['Number'])

# No more NaNs at all !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 9) Setting a baseline score using XGBoostClassifier:
# 0.8836965454744723 - cross_validation roc_auc_score:
# (0.8920077901764556, 0.8031278748850046) (roc_auc_score, accuracy)
# comp: 0.79003



# 10) Feature Engineering --------------------------------------------------------------------------------------------

# Visualizing correlations between feature and being transported to other dimension
#print(train_df.columns)
features = ['HomePlanet','CryoSleep', 'Destination', 'VIP','Deck','Side']
label = 'Transported'
to_bin_features = ['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck', 'Age', 'TotalExpenses']

# I (with a bit of help from GPT) came up with an excellent bar plot. I have not seen on internet anything so neat and
# effective, therefore it deserved to be a function xD Possibly helpful in the future.
def correlation_BarPlot(features, label):
    for feature in features:
        print(train_df.groupby([feature, label]).size())
        grouped = train_df.groupby([feature, label]).size()
        ax = grouped.unstack().plot(kind='bar', stacked=True, figsize=(12, 7), rot=0)

        # Iterate over each feature value
        for i, (index, row) in enumerate(grouped.unstack().iterrows()):
            # Number of appearances of particular value in feature
            total_count = row.sum()


            # Iterate over each value of 'Transported'
            for j, value in enumerate(row):
                percentage = (value / total_count) * 100


                # Middle of the part of the bar for visualization
                bar_bottom = 0 if j == 0 else row.iloc[:j].sum()
                bar_middle = bar_bottom + value / 2

                # Adding text on the part of the bar
                ax.text(i, bar_middle, f'{value} ({percentage:.2f}%)', ha='center', va='center')

        ax.xaxis.set_label_position('top')
        plt.xlabel(feature, fontsize=18)
        plt.xticks(rotation=360)
        plt.show()

correlation_BarPlot(features, label)

#---------------------------------------------------------------------------------------------------------------------
for feature in ['VIP', 'CryoSleep','Transported']:
    train_df[feature] = train_df[feature].apply(str)

train_df.replace({"VIP": {'True': 1,'False': 0},
                  "CryoSleep": {'True': 1, 'False': 0},
                  "Transported": {'True': 1,'False': 0}}, inplace=True, regex=True)

for feature in ['VIP', 'CryoSleep']:
    test_df[feature] = test_df[feature].apply(str)

test_df.replace({"VIP": {'True': 1,'False': 0},
                  "CryoSleep": {'True': 1, 'False': 0}}, inplace=True, regex=True)

# 10a) Deck + Side:
train_df['Deck_Side'] = train_df['Deck'] + train_df['Side']
test_df['Deck_Side'] = test_df['Deck'] + test_df['Side']


#  10b) Binned Number(binned by myself) (as a single feature not helpful)
train_df['Bin_Number'] = pd.cut(train_df['Number'], bins=np.arange(0,2200,200),
                                labels=np.arange(1,11,1), include_lowest =True)
test_df['Bin_Number'] = pd.cut(test_df['Number'], bins=np.arange(0,2200,200),
                               labels=np.arange(1,11,1), include_lowest =True)
train_df['Bin_Number'] = train_df['Bin_Number'].astype(str)
test_df['Bin_Number'] = test_df['Bin_Number'].astype(str)


# 10c) Deck + Bin_Number  (this helps)
train_df['Deck_BNumber'] = train_df['Deck'] + train_df['Bin_Number']
test_df['Deck_BNumber'] = test_df['Deck'] + test_df['Bin_Number']


# 10d) Side + Number (doesn't help)
# train_df['Side_BNumber'] = train_df['Side'] + train_df['Bin_Number']
# test_df['Side_BNumber'] = test_df['Side'] + test_df['Bin_Number']


# 10e) Deck + Side + Bin_Number (doesn't help)
# train_df['Deck_Side_BNumber'] = train_df['Deck_Side'] + train_df['Bin_Number']
# train_df['Deck_Side_BNumber'] = train_df['Deck_Side'] + train_df['Bin_Number']
# train_df.drop('Bin_Number', axis=1, inplace=True)
# test_df.drop('Bin_Number', axis=1, inplace=True)


# 10f) Has_Expenses + Has_x..  (doesn't help) and Transactions number
# train_df['Has_RoomService'] = np.where((train_df['RoomService'] > 0), 1, 0)
# test_df['Has_RoomService'] = np.where((test_df['RoomService'] > 0), 1, 0)
#
# train_df['Has_FoodCourt'] = np.where((train_df['FoodCourt'] > 0), 1, 0)
# test_df['Has_FoodCourt'] = np.where((test_df['FoodCourt'] > 0), 1, 0)
#
# train_df['Has_ShoppingMall'] = np.where((train_df['ShoppingMall'] > 0), 1, 0)
# test_df['Has_ShoppingMall'] = np.where((test_df['ShoppingMall'] > 0), 1, 0)
#
# train_df['Has_Spa'] = np.where((train_df['Spa'] > 0), 1, 0)
# test_df['Has_Spa'] = np.where((test_df['Spa'] > 0), 1, 0)
#
# train_df['Has_VRDeck'] = np.where((train_df['VRDeck'] > 0), 1, 0)
# test_df['Has_VRDeck'] = np.where((test_df['VRDeck'] > 0), 1, 0)
#

# No_Expenses looks like helping sometimes
# train_df['No_Expenses'] = np.where((train_df['TotalExpenses'] > 0), 0, 1)
# test_df['No_Expenses'] = np.where((test_df['TotalExpenses'] > 0), 0, 1)

train_df['Transaction_number'] = train_df.loc[:, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(np.count_nonzero, axis=1)
test_df['Transaction_number'] = test_df.loc[:, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(np.count_nonzero, axis=1)


# 10g) Group_Size  - doesn't help (xgb, cat)
#print(train_df.groupby(['Ticket'])['Ticket'].count())
# train_df['Group_Size'] = train_df.groupby(['Ticket'])['Ticket'].transform('count')
# # train_df['Group_Size'] = np.where((train_df['Group_Size'] == 1), 0 , train_df['Group_Size'] )
# # train_df['Group_Size'] = np.where((train_df['Group_Size'] > 1), 1 , train_df['Group_Size'] )
# test_df['Group_Size'] = test_df.groupby(['Ticket'])['Ticket'].transform('count')
# # test_df['Group_Size'] = np.where((test_df['Group_Size'] == 1), 0 , test_df['Group_Size'] )
# # test_df['Group_Size'] = np.where((test_df['Group_Size'] > 1), 1 , test_df['Group_Size'] )
# train_df['Group_Size'] = train_df['Group_Size'].astype(str)
# test_df['Group_Size'] = test_df['Group_Size'].astype(str)
#
#
# #10h) Group_Number, marking every specific group on the same ticket -----(doesn't help) (xgb, cat)
# counter = 1
# temp_counter = 0
# train_df['Group_Number'] = np.zeros(8693)
# train_df['Group_Size'] = train_df['Group_Size'].astype(int)
#
# for i, size in enumerate(train_df['Group_Size']):
#     try:
#         check8 = train_df.loc[i][14]
#         check9 = train_df.loc[i - 1][14]
#         check10 = train_df.iloc[i][22]
#         check11 = train_df.iloc[i-1][22]
#     except:
#         pass
#     if i > 0 and check8 == check9 and size > 1:
#         temp_counter += 1
#         train_df['Group_Number'] = np.where((train_df['Ticket'] == check8),counter, train_df['Group_Number'])
#         train_df.iloc[i-1][22] = np.where(train_df.iloc[i-1][22], counter, 0)
#         train_df.iloc[i][22] = np.where(train_df.iloc[i][22], counter, 0)
#     if i > 0 and check8 != check9 and temp_counter > 0:
#         temp_counter = 0
#         counter += 1
#
# train_df['Group_Number'] = train_df['Group_Number'].astype(str)
#
# counter = 1
# temp_counter = 0
# test_df['Group_Number'] = np.zeros(4277)
# test_df['Group_Size'] = test_df['Group_Size'].astype(int)
#
# for i, size in enumerate(test_df['Group_Size']):
#     try:
#         check8 = test_df.loc[i][13]
#         check9 = test_df.loc[i - 1][13]
#         check10 = test_df.iloc[i][22]
#         check11 = test_df.iloc[i-1][22]
#     except:
#         pass
#     if i > 0 and check8 == check9 and size > 1:
#         temp_counter += 1
#         test_df['Group_Number'] = np.where((test_df['Ticket'] == check8),counter, test_df['Group_Number'])
#         test_df.iloc[i-1][22] = np.where(test_df.iloc[i-1][22], counter, 0)
#         test_df.iloc[i][22] = np.where(test_df.iloc[i][22], counter, 0)
#     if i > 0 and check8 != check9 and temp_counter > 0:
#         temp_counter = 0
#         counter += 1
#
# test_df['Group_Number'] = test_df['Group_Number'].astype(str)
# print(train_df.head(10))


# # 10i) Solo =0, when ticket and surname the same and CryoSleep == 0, otherwise 1. Doesn't help (with or without Cryo).
#
#
# train_df['Solo'] = np.ones(8693)
# train_df['Solo'] = train_df['Solo'].astype(int)
#
# for i, size in enumerate(train_df['Solo']):
#     try:
#         check8 = train_df.loc[i][14]
#         check9 = train_df.loc[i - 1][14]
#         check10 = train_df.iloc[i][12].split(' ')[-1]
#         check11 = train_df.iloc[i-1][12].split(' ')[-1]
#         check12 = train_df.iloc[i][2]
#         check13 = train_df.iloc[i - 1][2]
#     except:
#         pass
#     if i > 0 and check8 == check9 and check10 == check11 and check12 == 0 and check13 == 0:
#         train_df._set_value((i-1),'Solo', 0)
#         train_df._set_value(i,'Solo', 0)
#
# print(test_df.shape)
# test_df['Solo'] = np.ones(4277)
# test_df['Solo'] = test_df['Solo'].astype(int)
#
# for i, size in enumerate(test_df['Solo']):
#     try:
#         check8 = test_df.loc[i][13]
#         check9 = test_df.loc[i - 1][13]
#         check10 = test_df.iloc[i][11].split(' ')[-1]
#         check11 = test_df.iloc[i-1][11].split(' ')[-1]
#         check12 = train_df.iloc[i][2]
#         check13 = train_df.iloc[i - 1][2]
#     except:
#         pass
#     if i > 0 and check8 == check9 and check10 == check11 and check12 == 0 and check13 == 0:
#         test_df._set_value((i-1),'Solo', 0)
#         test_df._set_value(i,'Solo', 0)

# 10j) Route - (xgb - nope; cat - yep). Complicated
# train_df['Route'] = train_df['HomePlanet'] + train_df['Destination']
# test_df['Route'] = test_df['HomePlanet'] + test_df['Destination']

# 10k) Age binning (complicated..)
# train_df['Age'] = pd.cut(train_df['Age'], bins=[0,18,25,150],
#                                 labels=[0, 18, 25], include_lowest =True)
# test_df['Age'] = pd.cut(test_df['Age'],  bins=[0,18,25, 150],
#                                 labels=[0, 18, 25], include_lowest =True)
# train_df['Age'] = train_df['Age'].astype(str)
# test_df['Age'] = test_df['Age'].astype(str)


# # 10l) Grouping expenses on Luxury and Necessities(idea from other notebook) and subtracting them creating another feature
# classy = ['RoomService', 'Spa', 'VRDeck']
# boring = ['FoodCourt', 'ShoppingMall']
#
# # Add expenses from RoomService, Spa, VRDeck
# train_df["Luxury"] = train_df[classy].sum(skipna=True, axis=1)
# test_df["Luxury"] = test_df[classy].sum(skipna=True, axis=1)
#
# # Add expenses from FoodCourt, ShoppingMall
# train_df["Necessities"] = train_df[boring].sum(skipna=True, axis=1)
# test_df["Necessities"] = test_df[boring].sum(skipna=True, axis=1)
#
# # Subtract these features
# train_df["Nec-Lux"] = train_df["Necessities"] - train_df["Luxury"]
# test_df["Nec-Lux"] = test_df["Necessities"] - test_df["Luxury"]
#
# #
# # # 10m) Creating Cabin region feature and one-hot encoding it (idea from other notebook)
# train_df['Cabin_region1']=(train_df['Number']<300).astype(int)
# train_df['Cabin_region2']=((train_df['Number']>=300) & (train_df['Number']<600)).astype(int)
# train_df['Cabin_region3']=((train_df['Number']>=600) & (train_df['Number']<900)).astype(int)
# train_df['Cabin_region4']=((train_df['Number']>=900) & (train_df['Number']<1200)).astype(int)
# train_df['Cabin_region5']=((train_df['Number']>=1200) & (train_df['Number']<1500)).astype(int)
# train_df['Cabin_region6']=((train_df['Number']>=1500) & (train_df['Number']<1800)).astype(int)
# train_df['Cabin_region7']=(train_df['Number']>=1800).astype(int)
#
#
# test_df['Cabin_region1']=(test_df['Number']<300).astype(int)
# test_df['Cabin_region2']=((test_df['Number']>=300) & (test_df['Number']<600)).astype(int)
# test_df['Cabin_region3']=((test_df['Number']>=600) & (test_df['Number']<900)).astype(int)
# test_df['Cabin_region4']=((test_df['Number']>=900) & (test_df['Number']<1200)).astype(int)
# test_df['Cabin_region5']=((test_df['Number']>=1200) & (test_df['Number']<1500)).astype(int)
# test_df['Cabin_region6']=((test_df['Number']>=1500) & (test_df['Number']<1800)).astype(int)
# test_df['Cabin_region7']=(test_df['Number']>=1800).astype(int)
#
# # 10n) Apply log transform (I have tried boxcox transformation before, but without visible gains) - effect not obvious.
# for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','TotalExpenses']:
#     train_df[col]=np.log(1+train_df[col])
#     test_df[col]=np.log(1+test_df[col])

#print(train_df.head(200))
# train_df.groupby(['Transported','Deck', pd.cut(train_df['Number'], np.arange(0, 2000, 200))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
# plt.xticks(rotation=30)
# plt.show()
# train_df.groupby(['Transported','Deck', 'Bin_Number']).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
# plt.xticks(rotation=30)
# plt.show()
# train_df.groupby(['Transported', 'Deck','Side']).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
# plt.show()

# train_df['Ticket'] = train_df['Ticket'].astype(int)
# test_df['Ticket'] = test_df['Ticket'].astype(int)
# print(train_df.head(20))
# print(train_df.dtypes)
# print(test_df.head(20))
# print(test_df.dtypes)

# 11) Droping useless features ----------------------------------------------------------------------------------------

y = train_df.Transported
train_df.drop(['PassengerId', 'Transported','Ticket' ,'Cabin', 'Name', 'Bin_Number'], axis=1, inplace=True)
X = train_df
test_df.drop(['PassengerId','Ticket', 'Cabin', 'Name', 'Bin_Number'], axis=1, inplace=True)
X_comp = test_df

# (from FE part) Removing skewness from numeric columns (no help) ------------------------------------------------------

# from scipy.stats import skew, boxcox_normmax
# from scipy.special import boxcox1p
# #print(X.dtypes,"--------")
# numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# numerics2 = []
#
# for i in X.columns:
#     if X[i].dtype in numeric_dtypes:
#         numerics2.append(i)
#
# skew_features = X[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
# #print(skew_features, "skew feat")
# high_skew = skew_features[skew_features > 0.5]
# skew_index = high_skew.index
# #print(skew_index, "skew index")
#
# for i in skew_index:
#     #fig, ax = plt.subplots(1, 3)
#     #sns.histplot(X[i], ax=ax[0])
#     #print(boxcox_normmax((X[i] + 1), brack=(-1.9, 2.0),  method='mle'))
#     fit_lambda = boxcox_normmax((X[i] + 1), brack=(-1.9, 2.0),  method='mle')
#     X[i] = boxcox1p(X[i], fit_lambda)
#     X_comp[i] = boxcox1p(X_comp[i], fit_lambda)
#     # (optional) plot train & test
#
#     #sns.histplot(X[i], ax=ax[1])
#     #sns.histplot(X_comp[i], ax=ax[2])
#     #plt.show()

# 12) Creating dummies -------------------------------------------------------------------------------------------------
X_dummies = pd.get_dummies(X)
X_dummies.replace({True: 1, False: 0}, inplace=True)
X_comp_dummies = pd.get_dummies(X_comp)
X_comp_dummies = X_comp_dummies.reindex(columns = X_dummies.columns, fill_value=0)
X_comp_dummies.replace({True: 1, False: 0}, inplace=True)


# 13) Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, train_size=0.9, random_state=42)


# 14) Testing models. Hyperparameter tuning with optuna.
def objective(trial, model_selector):
    if model_selector == 1:
        #xgbc
        # param = {
        #         "gamma": trial.suggest_int("gamma",0,0),
        #         "colsample_bytree": trial.suggest_float("colsample_bytree",0,1),
        #         "min_child_weight": trial.suggest_int("min_child_weight",0,15),
        #         "max_depth": trial.suggest_int("max_depth",0,15),
        #         "n_estimators": trial.suggest_int("n_estimators",1000,4000),
        #         "alpha": trial.suggest_float("alpha",0.00001,75),
        #         "learning_rate": trial.suggest_float("learning_rate",0.001,1),
        #         "colsample_bylevel": trial.suggest_float("colsample_bylevel",0,1),
        #         "colsample_bynode": trial.suggest_float("colsample_bynode",0,1),
        #         "random_state": trial.suggest_int("random_state",0,0),
        #         "subsample": trial.suggest_float("subsample",0,1),
        #         "lambda": trial.suggest_float("lambda", 0.001, 75)
        #     }
        param = {
            "gamma": trial.suggest_int("gamma", 0, 0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 8),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "n_estimators": trial.suggest_int("n_estimators", 1500, 2500),
            "alpha": trial.suggest_float("alpha", 1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0, 1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0, 1),
            "random_state": trial.suggest_int("random_state", 0, 0),
            "subsample": trial.suggest_float("subsample", 0.8, 1),
            "lambda": trial.suggest_float("lambda", 1, 75)
        }
        model = make_pipeline(XGBClassifier(**param))

    elif model_selector == 2:
        #lgbm
        param = {
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
            "max_depth": trial.suggest_int("max_depth", 0, 20),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0, 1),
            "random_state": trial.suggest_int("random_state", 0, 0),
            "num_leaves": trial.suggest_int("num_leaves",2,50),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0,1),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 8),
            "bagging_seed": trial.suggest_int("bagging_seed", 0, 8),
            "feature_fraction_seed": trial.suggest_int("feature_fraction_seed", 0, 8),
            "verbose":  trial.suggest_int("verbose",-1,-1)
        }

        model = make_pipeline(LGBMClassifier(**param))

    elif model_selector == 3:
        #rf
        param = {'n_estimators': trial.suggest_int("n_estimators", 1000, 2000),
              "max_depth": trial.suggest_int("max_depth", 1, 30),
              "max_samples": trial.suggest_float("max_samples", 0.4, 1),
              "max_features": trial.suggest_int("max_features", 1,40),
              "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
              "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5)

               }

        model = make_pipeline(RandomForestClassifier(**param))

    elif model_selector == 4:
        #ada
        param = {'n_estimators': trial.suggest_int("n_estimators", 50, 1500),
              "learning_rate": trial.suggest_float("learning_rate", 0.001, 2),
              "algorithm": trial.suggest_categorical("algorithm", ['SAMME', 'SAMME.R'])
               }
        model = make_pipeline(AdaBoostClassifier(**param))

    elif model_selector == 5:
        # catboost
        param = {
            #"objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),



        }
        model = make_pipeline( RobustScaler(), CatBoostClassifier(**param, silent=True))

    results = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=5)
    result = results.mean()

    return result

models = {'xgbc': 1, 'lgbm': 2, 'rf': 3, 'ada':4, 'cat': 5}



# study = opt.create_study(direction='maximize')
# study.optimize(lambda trial: objective(trial, models['rf']), n_trials=30, n_jobs=-1)
# print(study.best_trial.params)
# print(study.best_value)
# results = (study.best_trial.params, study.best_value)
# with open(f'cat02.json', 'w') as json_file:
#     json.dump(results, json_file)
# print(opt.visualization.is_available())

# 15) Best parameters for every model.
#best so far - pierwszy od góry
xgbc_params = {'gamma': 0, 'colsample_bytree': 0.6668197838452801, 'min_child_weight': 4, 'max_depth': 11,
               'n_estimators': 2119, 'alpha': 1.8046734104101436, 'learning_rate': 0.03564992496627689,
               'colsample_bylevel': 0.3891993131985869, 'colsample_bynode': 0.4103108111735978, 'random_state': 0,
               'subsample': 0.9277986532925703, 'lambda': 44.87436571119498}

catboost_params = {'learning_rate': 0.02156513983572557, 'colsample_bylevel': 0.08167978252292739, 'max_depth': 12,
                   'boosting_type': 'Ordered', 'min_data_in_leaf': 72}
lgbm_params = {'colsample_bytree': 0.8996458099506537, 'max_depth': 6, 'n_estimators': 3829, 'reg_alpha': 1.974253940612134,
               'reg_lambda': 0.10029650121603073, 'learning_rate': 0.0027087468294310635, 'colsample_bynode': 0.3250556026706851,
               'random_state': 0, 'num_leaves': 22, 'bagging_fraction': 0.988940300870413, 'bagging_freq': 0, 'bagging_seed': 0,
               'feature_fraction_seed': 5, 'verbose': -1}
rf_params = {'n_estimators': 1487, 'max_depth': 9, 'max_samples': 0.7818124889495328, 'max_features': 39,
             'min_samples_split': 5, 'min_samples_leaf': 3}
ada_params = {'n_estimators': 922, 'learning_rate': 1.1705013045122792, 'algorithm': 'SAMME'}


# 16) Creating final pipelines
xgbc_pipe = make_pipeline(RobustScaler(), XGBClassifier(**xgbc_params))
catboost_pipe = make_pipeline(  RobustScaler(), CatBoostClassifier(**catboost_params, silent=True))
lgbm_pipe = make_pipeline(RobustScaler(), LGBMClassifier(**lgbm_params))
rf_pipe = make_pipeline(RobustScaler(), RandomForestClassifier(**rf_params))
ada_pipe = make_pipeline(RobustScaler(), AdaBoostClassifier(**ada_params))

# 17) Checking final performance on test set, and saving results
def final_test(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    rc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,-1])
    accuracy = model.score(X_test, y_test)
    with open(f'cat02.json', 'w') as json_file:
        json.dump(results, json_file)
    return rc_score, accuracy


def results(model, X, y, X_comp):
    model.fit(X, y)
    preds = model.predict_proba(X_comp)
    print(type(preds))
    print(preds[:10])
    np.savetxt('huh.txt', preds)
    preds = model.predict(X_comp)
    print(test_df_copy.head())
    preds = np.array(preds, dtype=bool)
    result = pd.DataFrame({'PassengerId': test_df_copy.PassengerId,
                           'Transported': preds.squeeze()})
    print(result.head(15))
    result.to_csv('results_spaceship.csv', index=False)


models = [xgbc_pipe, catboost_pipe, lgbm_pipe, rf_pipe, ada_pipe]

# Manually saving is faster, and more reliable for every model
final_test(catboost_pipe, X_train, X_test, y_train, y_test)
results(catboost_pipe, X_dummies, y, X_comp_dummies)


# 18. Catboost offers the best performance: cross_validation roc_auc_score: 0.9054885156619108.
# Strangly hypertuned with: 40 trials, split 0.9, luxury, cabin region, log,
# transaction number, rando_State = 42, but later trained on dataset without luxury, cabin region and log, gives the best
# competition predictions: 0.80009.

cat2 = np.loadtxt('cat_best.txt')
preds = np.array(cat2, dtype=bool)
result = pd.DataFrame({'PassengerId': test_df_copy.PassengerId,
                           'Transported': preds.squeeze()})
result.to_csv('results.csv', index=False)




