#!/usr/bin/env python
# coding: utf-8

# # load csv file

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('nBagg')
import os
os.chdir('F:\internship')
import pandas as pd


# In[2]:


A=pd.read_csv('zomato.csv')
data=pd.DataFrame(A)
data


# # Data Preprocessing and EDA
# as we no need of adrdress,phone,url column because it indicates no value or gives same information for modelling we look for location only

# In[3]:


data.shape
# In[4]:
drop_col=['url','phone','address', 'listed_in(city)']
data.drop(drop_col,axis=1,inplace=True)
# In[5]:
data.duplicated().sum()
# In[6]:
data.drop_duplicates(inplace=True)
# In[7]:
data.duplicated().sum()
# In[8]:
data.index

# Changing columns name as per our convinence
data=data.rename(columns={'cost(two)':'cost','listed_in(type)':'listed_type'})
data
# In[13]:
data.votes.describe()


# there are some restaurent with 0 votes too good insights is it!
# and there is a restaurent with 16832 votes too

# In[14]:
print(data.shape)
data.isnull().sum()
# In[15]:
data['rate'].dtype
# In[16]:
data['rate'].unique()
# In[17]:
data['rate']=data['rate'].replace('NEW',np.NaN)
data['rate']=data['rate'].replace('-',np.NaN)
data['rate'].unique()


# In[18]:


data.rate=data.rate.astype(str)
data.rate=data.rate.apply(lambda x : x.replace('/5',''))
data.rate=data.rate.astype(float)


# we can see the rate column is in str type which is non supported type for ML training so we change into float as per the rating method  and need to convert all catogorical column into numerical we will do at time of modelling after visualization for insights

#  so we replaced incorrect value and symbols in rate with NaN value using Replace method

# In[19]:


data['rate'].dtype


# In[20]:


data['cuisines'].isna().sum()


# In[21]:


data.rest_type.value_counts()


# In[22]:


data['rest_type'].isnull().sum()


# In[23]:


data['rest_type'].fillna(value="Quick Bites",inplace=True)


# replacing missing values with most occuring value  in rest_type

# In[24]:


data['cuisines'].value_counts()


# In[184]:


data['cuisines'].fillna('North Indian'or 'North Indian ,Chinese' or 'South Indian' ,inplace =True)


# replacing missing values with most occuring value  in cuisines

# In[25]:


data.isna().sum()


# lets reduce the null values of remaining features using replacing means of others or dropping minimal values

# In[26]:


data['reviews_list'].dtype


# In[27]:


data.reviews_list.values[:1]


# In[30]:


#we could extract these values from reviews and take their mean to fill rate column
data.reviews_list.values[1]


# In[28]:


type(data.reviews_list[1])


# In[29]:


import ast
ast.literal_eval(data.reviews_list.values[1])


# In[30]:


get_ipython().run_line_magic('time', 'data.reviews_list=data.reviews_list.apply(lambda x: ast.literal_eval(x))')


# In[31]:


type(data.reviews_list[100])


# In[32]:


data.reviews_list[0][0][0].split()[0]


# In[33]:


data['reviews_list'].values[1]


# In[34]:


def extract_from_review_list(x):
    #extract the rate value out of a string inside tuple
    # ensure that x is not Null and there is more than one rate
    if not x or len(x) <= 1:
        return None
    rate_new= [float(i[0].replace('Rated','').strip())  for i in x if type(i[0])== str]
    return round((sum(rate_new)/len(rate_new)),1)


# In[35]:


get_ipython().run_line_magic('time', "data['rate_new']=data.reviews_list.apply(lambda x: extract_from_review_list(x))")


# In[36]:


data.loc[:,['rate','rate_new']].sample(10)


# In[37]:


# apply the changes
nan_index = data.query('rate != rate & rate_new == rate_new').index
for i in nan_index:
    data.loc[i,'rate'] = data.loc[i,'rate_new']


# In[38]:


nan_index


# In[39]:


data.rate.isna().sum()


# In[41]:


data.dropna(subset=['rate', 'cost(2)'],inplace=True)


# In[42]:


data.drop(['rate_new'],axis=1,inplace=True)


# In[83]:


data.isna().sum()


# we finllay reduced to zero of null values mostly column 'rate' is filled with mean of review_list for null values 

# 1.reduce the biggest null values of dish_liked with extracting foods stated by reviewrs in reviews_last because its nearly half of the rows filled with null if we remove it is hard for good prediction \
# 2.otherwise fill with most occuring values but it will become bias where null values is huge

# In[44]:


data.dish_liked


# In[46]:


#converting to lowercase
data.dish_liked=data.dish_liked.apply(lambda x:x.lower().strip() if isinstance(x,str) else x)


# In[47]:


data.dish_liked.head(10)


# as we can see dishes liked or disliked are mentioned in reviews so if we can extract these dishes we can fill the nan values of dish_liked column
# we will start by getting a list of all the dishes available from our dataset

# In[48]:


dish_list=[]
for i in list(data.index):
        #print(type(data.dish_liked[i])) #checking whether it shows  index and dish_name type
        if data.dish_liked[i]!='NaN' and isinstance(data.dish_liked[i],str):
            k=data['dish_liked'][i].split(',')
            dish_list.extend(k)
print(dish_list)


# In[49]:


len(dish_list)


# In[50]:


dish_list=set(dish_list) #getting unique dishes values


# In[51]:


len(dish_list)


# In[52]:


p=data.reviews_list[0]
' '.join([i[1].replace('RATED\n ','') for i in p]).replace('\n','').replace('\S+','').replace('?','').replace('Ã','').replace('\\x','').strip().lower()


# In[53]:


# clear the text
def clear_text(t):
    '''
    clear the input text t
    '''
    return ' '.join([i[1].replace("RATED\n  ",'') for i in t]).encode('utf8').decode('ascii',errors='replace'). replace('?','').replace('�','').replace('\n','').replace('.',' ').strip().lower()


# In[54]:


data['reviews_text'] = data.reviews_list.apply(lambda x: clear_text(x))


# In[55]:


dish_list.intersection(data['reviews_text'][100].split())


# In[56]:


data['dish_liked_new']=data.reviews_text.apply(lambda x: ', '.join(list(dish_list.intersection(x.split()))))


# In[57]:


data.dish_liked_new.isna().sum()


# In[58]:


# get sample to compare
data.query('dish_liked != dish_liked')[['dish_liked','dish_liked_new']].sample(5,random_state=1)


# In[59]:


nan_index=data.query('dish_liked !=dish_liked & dish_liked_new==dish_liked_new').index


# In[60]:


get_ipython().run_cell_magic('time', '', "for i in nan_index:\n    data.loc[i,'dish_liked']=data.loc[i,'dish_liked_new'] ")


# In[61]:


data.drop(['dish_liked_new','reviews_text'],axis=1,inplace=True)


# In[84]:


data.isna().sum()


# finally we got them to zero null values let do some binary opertion for simple catogorical values now later we will do encoding

# # simple encoding for simple features

# In[63]:


data['online_order']=pd.get_dummies(data.online_order,drop_first=True)
data['book_table']=pd.get_dummies(data.book_table,drop_first=True)


# In[246]:


data['cost']=data['cost'].astype(str)
data['cost']=data['cost'].apply(lambda v:v.replace(',',''))
data['cost']=data['cost'].astype(float)


# In[86]:


data['cost(two)'].dtype
data['votes'].dtype


# changing type of variables to numerical 

# In[247]:


data


# # INSIGHTS AND VISUALIZATION (using matplotlib,seaborn) OF OUR DATASETS 

# In[68]:


data['book_table'].value_counts()


# In[69]:


sns.countplot(data.book_table,palette = "Set1")
plt.title('No of tables booked and not booked  0=No 1=Yes ')
plt.show


# In[70]:


data['online_order'].value_counts()


# In[382]:


sns.countplot(data.online_order,palette = "dark:salmon_r")
plt.title('count of Online order  0=No 1=Yes ')
plt.show


# RESTAURANTS NAME AND ITS COUNTS 

# In[72]:


data['name'].value_counts()


# In[73]:


plt.figure(figsize=(10,5))
data.name.value_counts()[:10].plot(kind='bar',color='black')
plt.title("Top 10 Restaurants and Names",weight='bold')
plt.ylabel('count')
plt.show()


# INSIGHTS OF LOCATIONS AND RESTAURANTS 

# In[116]:


data['location'].value_counts()


# In[131]:


plt.figure(figsize=(15,15))
ax =data.location.value_counts()[:15].plot(kind='bar',color='c')
plt.title('Number of Restaurants in each location top 15', weight='bold',)
plt.xlabel('location')
plt.ylabel('No. of Restaurants')
plt.show()


# In[220]:


plt.figure(figsize=(12,7))
colors = ['gold', 'red', 'lightcoral', 'lightskyblue','blue','green','silver','violet']
explode = (0.4, 0, 0, 0,0,0,0,0,0,0)  # explode 1st slice
ax=data.location.value_counts()[:10].plot(kind='pie',explode=explode,shadow=True,colors=colors,startangle=140,autopct='%1.1f%%')
plt.axis('equal')
plt.title("Percentage of restaurants present in that location", weight = 'bold')


# RESTAURANTS TYPE AND COUNTS 

# In[83]:


data['rest_type'].value_counts()


# In[189]:


plt.figure(figsize=(12,9))
colors = ['purple','brown', 'red', 'lightcoral', 'lightskyblue','blue','green','silver','pink','yellow']
ax=data.rest_type.value_counts()[:10].plot(kind='pie',colors=colors,startangle=180,autopct='%1.1f%%')
plt.axis('equal')
plt.title("Percentage of restaurants_type", weight = 'bold')


# In[249]:


M=data['cost'].max()
m=data['cost'].min()
a=data['cost'].mean()
print('maxiumum cost is {0} , mimimum cost is {1} , average cost per two is {2}'.format((M),(m),(a)))


# Listed_type is visualized using treemap for better visual of counts and understadings

# In[129]:


plt.figure(figsize=(12,7))
data.listed_type.value_counts().plot(kind='bar')
plt.show()


# In[227]:


data['listed_type'].value_counts()


# In[274]:


import squarify 
plt.figure(figsize=(10,5))
squarify.plot(sizes=data.listed_type.value_counts(),label=['Delivery',
'Dine-out',
'Desserts',     
'Cafes',  
'Drinks & nightlife',   
'Buffet',                  
'Pubs and bars'],alpha=0.5, norm_x=100)
plt.show()


# In[379]:


plt.figure(figsize=(5,5))
sns.distplot(fdata['cost'],color='blue')
plt.show()


# lets check Normal distribution of our label(y)

# In[75]:


data.rate.hist(color='b',bins=30)
plt.axvline(x=data.rate.mean(),color='red',ls='--')
plt.title("Restaurant's ratings")
plt.xlabel('Rating')
plt.ylabel('No. of Restaurants')
plt.show()
print(data.rate.mean())


# Lets go into little good Insights

# In[250]:


data['cost'].value_counts()


# In[251]:


plt.figure(figsize=(15,8))
data['cost'].value_counts().plot(kind='bar')
plt.title('Average cost for two person(in %) ', weight='bold')
plt.xlabel('Cost')
plt.ylabel('count')
plt.show()
    
    


# In[236]:


fig=plt.figure(figsize=(10,5))
sns.pointplot(data['rest_type'][:20], data['rate']).set_title('Rate vs Restaurant type',weight='bold')
sns.set_theme(style='darkgrid')
plt.show()


# In[70]:


f,ax=plt.subplots(figsize=(16,10))
g=sns.pointplot(y='rate',x='rest_type',data=data)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
plt.title('Restaurant type vs Rate', weight = 'bold')
plt.show()


# Most favourite dishes in restaurants

# In[238]:


f,ax=plt.subplots(figsize=(12,8))
sns.countplot(x='rate',data=data,hue='online_order').set_title("rate vs online order",weight = 'bold')
plt.ylabel("Restaurants that Accept/Not Accepting online orders")
plt.show()


# In[246]:


plt.figure(figsize=(12,10))
data.cuisines.value_counts()[:10].plot(kind='bar',color='y').set_title('Top 10 cuisines in Bangalore',weight='bold')
plt.xlabel('cuisines type')
plt.ylabel('No of restaurants')


# # Feature Engineering (One Hot Encoding & Label Encoding) 

# LABEL ENCODING AND DROP UNREQUIRED COLUMNS FOR TRAINING

# In[252]:


fdata=data.copy()


# In[255]:


fdata.drop(['reviews_list','name'],axis=1,inplace=True)
fdata.drop(['dish_liked'],axis=1,inplace=True)
fdata.drop(['listed_type'],axis=1,inplace=True)


# In[256]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
list1=['rest_type','location','cuisines','menu_item']
for i in list1:
    fdata[i]=LE.fit_transform(fdata[i])
fdata.head()


# In[257]:


fdata.sample(10)


# WE JUST CHECK HOW BINARY ENCODING WORKS IN OUR DATASETS WE WONT USE FOR MODELS WE USING

# ONE HOT ENCODING occupies more columns and gives more dimensionality acess use i go with "BINARY ENCODING" which is mixture of hash encoder and one hot coding which takes lesser columns (converts cat into ordinal first then convert to numerical which is also nominal one) and also we have different catagories in our features columns

# In[258]:


set2=data.copy()


# In[259]:


set2.drop(['reviews_list','name'],axis=1,inplace=True)
set2.drop(['dish_liked'],axis=1,inplace=True)
set2.drop(['listed_type'],axis=1,inplace=True)


# In[260]:


set2


# BINARY ENCODING 

# In[102]:


from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols =['location'])
#transforming the column after fitting
one= encoder.fit_transform(set2['location'])
#concating
set2 = pd.concat([set2,one], axis = 1)


# In[ ]:


encoder = BinaryEncoder(cols =['rest_type'])
two= encoder.fit_transform(set2['rest_type'])
set2 = pd.concat([set2,two], axis = 1)


# In[ ]:


encoder = BinaryEncoder(cols =['cuisines'])
three= encoder.fit_transform(set2['cuisines'])
set2 = pd.concat([set2,three], axis = 1)


# In[ ]:


encoder = BinaryEncoder(cols =['menu_item'])
three= encoder.fit_transform(set2['menu_item'])
set2 = pd.concat([set2,three], axis = 1)


# In[106]:


set2.drop(['location','rest_type','cuisines'],inplace=True,axis=1)
set2.drop(['menu_item'],inplace=True,axis=1)


# In[107]:


set2.sample(10)


# # Data transformation and Features Scaling

# WE USING LABEL ENCODED FEATURES FOR OUR MODEL BUILDING

# In[108]:


fdata.rate.unique()


# In[109]:


set2.shape #using binary encoder


# In[110]:


fdata.shape #using label encoder


# In[262]:


Features=fdata.drop(['rate'],axis=1)
Features.shape


# In[263]:


Features.dtypes


# In[117]:


fdata['rate'].unique()


# In[118]:


label=fdata['rate'].values
label


# # Splitting Datasets

# Splitting the datasets into train and test data to see how our model performs to untrained datas
# train set = 80% and test set =20%

# In[264]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(Features,label,test_size=0.2,random_state=42)


# In[265]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# # REGRESSION MODELS

# # 1.linear regression

# In[267]:
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR
LR.fit(X_train,y_train)
print(LR.score(X_train,y_train))
print(LR.score(X_test,y_test))
print('intercept =',LR.intercept_)
print('coefficient =',LR.coef_)


# PREDICTION USING Linear Regression

# In[270]:


YP=LR.predict(X_test)
YP


# In[271]:


from sklearn.metrics import r2_score
LR=r2_score(y_test,YP)*100
print("Accuracy score for LR:",LR)


# Prediction is not up to the mark in linear regression as it score is just 25% 

# # 2. Random Forest Regressor

# In[272]:


from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(n_estimators=650,random_state=245,min_samples_leaf=.0001)
RF.fit(X_train,y_train)
RF_predict=RF.predict(X_test)


# In[273]:


print(RF.score(X_train,y_train))
print(RF.score(X_test,y_test))


# SAMPLE TESTING

# In[274]:


new=pd.DataFrame({'a':[1],'b':[0], 'c':[918],'d':[1],'e':[20],'f':[662],'g':[800.0],'h':[8519]})
predicted=RF.predict(new)
predicted


# In[275]:


sample=pd.DataFrame({"Actual Rating":y_test,
             "Predicted Rating":np.round(RF_predict,2)})
sample


# R^2 = (Y-Y^)^2 ------ (expected-predicted)^2

# In[276]:


from sklearn.metrics import r2_score
RFr2=r2_score(y_test,RF_predict)*100
print("Accuracy score for RF:",RFr2)


# # 3.Support Vector Machine

# In[284]:


from sklearn.svm import SVR
svr=SVR()
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',kernel='rbf', max_iter=-1, 
    shrinking=True, tol=0.001, verbose=False)
svr.fit(X_train,y_train)
print(svr.score(X_train,y_train)*100)
print(svr.score(X_test,y_test)*100)


# In[285]:


svr_pred=svr.predict(X_test)
svr_pred


# In[286]:
svr=r2_score(y_test,svr_pred)*100
print("Accuracy score for RF:",svr)


# Its look like SVM gives weak predictive model so dont worry we have many to go algorithms

# # BayesinRidge

# In[186]:


from sklearn import linear_model
BR = linear_model.BayesianRidge()
BR.fit(X_train,y_train)
BR.predict(X_test)


# In[201]:
bayesin=BR.predict(X_test)


# In[187]:
print(BR.score(X_train,y_train)*100)
print(BR.score(X_test,y_test)*100)

# In[280]:
br=r2_score(y_test,bayesin)*100
print("Accuracy score for RF:",br)


# # ExtraTree Regressor

# In[190]:
from sklearn.ensemble import  ExtraTreesRegressor
ETR=ExtraTreesRegressor(n_estimators = 150)
ETR.fit(X_train,y_train)
y_pd=ETR.predict(X_test)
y_pd

# In[193]:

ETR.score(X_test,y_test)
# In[199]:
EXR=r2_score(y_test,y_pd)*100
print("Accuracy score for RF:",EXR)

# ET Regressor is giving best than random forest as it is like same as random forest but it use whole training data set for trees and 
# uses randomly split unlike random forest of optimal split and its an ensemble technique bootstrap(bagging)

sample=pd.DataFrame({"Actual Rating":y_test,
             "Predicted Rating":np.round(y_pd,2)})
sample.sample(5)


from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pd)))
rmse


# # Ridge Regression

# In[200]:
from sklearn.linear_model import Ridge
rr=Ridge()
rr.fit(X_train,y_train)
print(rr.score(X_train,y_train))
print(rr.score(X_test,y_test))
rr_pred=rr.predict(X_test)
rr_score=r2_score(y_test,rr_pred)*100
print("Accuracy score for RidgeR :",rr_score)


# # DATA MODELS ACCURACIES

# In[361]:


Final_scores=pd.DataFrame({"Model Names":['Linear Reg','RandomForest Reg','Support Vector Reg','Bayesin Ridge','Extra tree Reg','Ridge Reg'],
            "Accuracy socre":[LR,RFr2,svr,br,EXR,rr_score]})
Final_scores


# # Checking multicolinearity  and correlation factors 

# In[354]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

X=fdata[['online_order','book_table','votes','location','rest_type','cuisines','cost','menu_item']]

vif= pd.DataFrame()
vif["features"] = X.columns
vif['VIF']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)


# In[296]:
corr=Features.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)


# # Saving Models
# Saving best model in pickle to use for further prediction
import pickle as pkl
# In[358]:
with open('RFmodel','wb') as file:
    pkl.dump(RF,file)
# In[359]:
with open('ETRmodel','wb') as file:
    pkl.dump(ETR,file)
# In[364]:
Final_scores.to_csv("Models Prediction Scores.csv",index=False)




