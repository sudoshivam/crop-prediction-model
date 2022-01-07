#Importing Liabraries

import numpy as np
import pandas as pd
import pickle

#For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#For interactivity
from ipywidgets import interact

#For warnings
import warnings
warnings.filterwarnings('ignore')

#For Clustering Analysis
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Loading Dataset
data = pd.read_csv('Crop_recommendation.csv')
data

#Shape of dataset
print("Shape of the dataset :",data.shape)

#Checking missing values
data.isnull().sum()

#Checking Crops present in Dataset
data['label'].value_counts()

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(), annot=True,cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')

#plt.title('Correlation between different features', fontsize = 15, c='black')
#plt.show()

#sns.pairplot(data,hue = 'label')

print("Average Ratio of nitrogen in the soil : {0: .2f}".format(data['N'].mean()))
print("Average Ratio of Phosphorous in the soil : {0: .2f}".format(data['P'].mean()))
print("Average Ratio of Potassium in the soil : {0: .2f}".format(data['K'].mean()))
print("Average temperature in Celsius : {0: .2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in % is : {0: .2f}".format(data['humidity'].mean()))
print("Average pH value of the soil : {0: .2f}".format(data['ph'].mean()))
print("Average Rain fall in mm : {0: .2f}".format(data['rainfall'].mean()))

@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    print("-------------------------------------------------")
    print("Statistics for Nitrogen :")
    print("Minimum Nitrogen Required :", x['N'].min())
    print("Average Nitrogen Required :", x['N'].mean())
    print("Maximum Nitrogen Required :", x['N'].max())
    print("-------------------------------------------------")
    print("Statistics for Phosphorous :")
    print("Minimum Phosphorous Required :", x['P'].min())
    print("Average Phosphorous Required :", x['P'].mean())
    print("Maximum Phosphorous Required :", x['P'].max())
    print("-------------------------------------------------")
    print("Statistics for Potassium :")
    print("Minimum Potassium Required :", x['K'].min())
    print("Average Potassium Required :", x['K'].mean())
    print("Maximum Potassium Required :", x['K'].max())
    print("-------------------------------------------------")
    print("Statistics for Temperature :")
    print("Minimum Temperature Required : {0: .2f}".format(x['temperature'].min()))
    print("Average Temperature Required : {0: .2f}".format(x['temperature'].mean()))
    print("Maximum Temperature Required : {0: .2f}".format(x['temperature'].max()))
    print("-------------------------------------------------")
    print("Statistics for Humidity :")
    print("Minimum Humidity Required : {0: .2f}".format(x['humidity'].min()))
    print("Average Humidity Required : {0: .2f}".format(x['humidity'].mean()))
    print("Maximum Humidity Required : {0: .2f}".format(x['humidity'].max()))
    print("-------------------------------------------------")
    print("Statistics for PH :")
    print("Minimum PH Required : {0: .2f}".format(x['ph'].min()))
    print("Average PH Required : {0: .2f}".format(x['ph'].mean()))
    print("Maximum PH Required : {0: .2f}".format(x['ph'].max()))
    print("-------------------------------------------------")
    print("Statistics for Rainfall :")
    print("Minimum Rainfall Required : {0: .2f}".format(x['rainfall'].min()))
    print("Average Rainfall Required : {0: .2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall Required : {0: .2f}".format(x['rainfall'].max()))
    print("-------------------------------------------------")

    
@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average Value for",conditions,"is {0: .2f}".format(data[conditions].mean()))
    print("----------------------------------------------------------------")
    print("Rice : {0: .2f}".format(data[(data['label'] == 'rice')][conditions].mean()))
    print("Black Grams : {0: .2f}".format(data[(data['label'] == 'blackgram')][conditions].mean()))
    print("Banana : {0: .2f}".format(data[(data['label'] == 'banana')][conditions].mean()))
    print("Jute : {0: .2f}".format(data[(data['label'] == 'jute')][conditions].mean()))
    print("Coconut : {0: .2f}".format(data[(data['label'] == 'coconut')][conditions].mean()))
    print("Apple : {0: .2f}".format(data[(data['label'] == 'apple')][conditions].mean()))
    print("Papaya : {0: .2f}".format(data[(data['label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0: .2f}".format(data[(data['label'] == 'muskmelon')][conditions].mean()))
    print("Grapes : {0: .2f}".format(data[(data['label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0: .2f}".format(data[(data['label'] == 'watermelon')][conditions].mean()))
    print("Kedney Beans : {0: .2f}".format(data[(data['label'] == 'kidneybeans')][conditions].mean()))
    print("Mung Beans : {0: .2f}".format(data[(data['label'] == 'mungbean')][conditions].mean()))
    print("Oranges : {0: .2f}".format(data[(data['label'] == 'orange')][conditions].mean()))
    print("Chick Peas : {0: .2f}".format(data[(data['label'] == 'chickpea')][conditions].mean()))
    print("Lentils : {0: .2f}".format(data[(data['label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0: .2f}".format(data[(data['label'] == 'cotton')][conditions].mean()))
    print("Maize : {0: .2f}".format(data[(data['label'] == 'maize')][conditions].mean()))
    print("Moth Beans : {0: .2f}".format(data[(data['label'] == 'mothbeans')][conditions].mean()))
    print("Pigeon peas : {0: .2f}".format(data[(data['label'] == 'pigeonpeas')][conditions].mean()))
    print("Mango : {0: .2f}".format(data[(data['label'] == 'mango')][conditions].mean()))
    print("Pomegrante : {0: .2f}".format(data[(data['label'] == 'pomegrante')][conditions].mean()))
    print("Coffee : {0: .2f}".format(data[(data['label'] == 'coffee')][conditions].mean()))

plt.figure(figsize=(15,8))
plt.subplot(2,4,1)
sns.distplot(data['N'],color = 'blue')
plt.xlabel('Ratio of Nitrogen',fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['P'],color = 'green')
plt.xlabel('Ratio of Phosphorous',fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['K'],color = 'darkblue')
plt.xlabel('Ratio of Potassium',fontsize = 12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['temperature'],color = 'black')
plt.xlabel('Temperature',fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(data['rainfall'],color = 'grey')
plt.xlabel('Rainfall',fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['humidity'],color = 'lightgreen')
plt.xlabel('Humidity',fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['ph'],color = 'darkgreen')
plt.xlabel('ph level',fontsize = 12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize = 20)
#plt.show()


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average",conditions,'\n')
    print(data[data[conditions] > data[conditions].mean()]['label'].unique())
    print("-------------------------------------------------------")
    print("Crops which require less than average",conditions,'\n')
    print(data[data[conditions] <= data[conditions].mean()]['label'].unique())

print("Crops which requires very High rainfall:",data[data['rainfall'] > 200]['label'].unique())
print("Crops which requires very Low rainfall:",data[data['rainfall'] < 40]['label'].unique())

print("Crops which requires very High ratio of Nitrogen Content in soil :",data[data['N'] > 120]['label'].unique())
print("Crops which requires very High ratio of Phosphorous Content in soil :",data[data['P'] > 100]['label'].unique())
print("Crops which requires very High ratio of Potassium Content in soil :",data[data['K'] > 200]['label'].unique())
print("Crops which requires very High Rainfall :",data[data['rainfall'] > 200]['label'].unique())
print("Crops which requires very Low Rainfall:",data[data['rainfall'] < 40]['label'].unique())
print("Crops which requires very Low Temperature :",data[data['temperature'] < 10]['label'].unique())
print("Crops which requires very High Temperature :",data[data['temperature'] > 40]['label'].unique())
print("Crops which requires very Low Humidity :",data[data['humidity'] < 20]['label'].unique())
print("Crops which requires very Low pH :",data[data['ph'] < 4]['label'].unique())
print("Crops which requires very High pH :",data[data['ph'] > 8]['label'].unique())

print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("--------------------------------------------------------------------------")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("--------------------------------------------------------------------------")
print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())

#Removing the Labels column 
x = data.drop(['label'], axis=1)

#Selecting all values of data
x = x.values

#Checking the shape
print(x.shape)

#Determining Optimum number of Clusters within Dataset by using K-means Clustering
plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
#Plotting the Results
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method',fontsize = 20)
plt.xlabel('No. of Cluster')
plt.ylabel('wcss')
#plt.show()

#Implementing K-means Algorithm to perform Clustering Analysis
km = KMeans(n_clusters = 4,init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#Lets find out results
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a],axis = 1)
z = z.rename(columns = {0: 'cluster'})

#Checking Clusters of Each crop
print("Checking results after applying K-means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Forth Cluster:", z[z['cluster'] == 3]['label'].unique())

#Splitting dataset for Predictive Modelling
y = data['label']
x = data.drop(['label'],axis = 1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

#Training and Testing Sets for Validation of Results
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

print("The shape of x train:", x_train.shape)
print("The shape of x test:", x_test.shape)
print("The shape of y train:", y_train.shape)
print("The shape of y test:", y_test.shape)


model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

filename = 'crop_pred_model'
pickle.dump(model, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[90, 40, 40, 20, 80, 7, 200]]))