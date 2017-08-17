import csv
import pandas as pd
import numpy as np
from sklearn import linear_model, cross_validation, svm, preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#read data
data1 = pd.read_csv('wheat-2013-supervised.csv')
data2 = pd.read_csv('wheat-2014-supervised.csv')


merged = data1.append(data2, ignore_index=True)
merged = merged[["CountyName","State","Latitude","Longitude","Date","apparentTemperatureMax","apparentTemperatureMin","cloudCover","dewPoint","humidity","precipIntensity","precipIntensityMax","precipProbability","precipAccumulation","precipTypeIsRain","precipTypeIsSnow","precipTypeIsOther",	"pressure",	"temperatureMax","temperatureMin","visibility",	"windBearing","windSpeed","NDVI","DayInSeason","Yield" ]]
merged.to_csv('merged.csv', index=None, header=True)
mg = pd.read_csv('merged.csv')

mg = mg[["Latitude","Longitude","apparentTemperatureMax","apparentTemperatureMin","cloudCover","dewPoint","humidity","precipIntensity","precipIntensityMax","precipProbability","precipAccumulation","precipTypeIsRain","precipTypeIsSnow","precipTypeIsOther",	"pressure",	"temperatureMax","temperatureMin","visibility",	"windBearing","windSpeed","NDVI","DayInSeason","Yield" ]]



mg.dropna(inplace=True)

X = np.array(mg.drop(["Yield"],1))

y = np.asarray(mg['Yield'], dtype="|S6")


# Code for regression
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)



# Code for decision tree and cross_validation
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=3,random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())
clf.fit(X,y)
print(clf.predict(X))

