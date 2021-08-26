#!/usr/bin/env python
# coding: utf-8

# #                                      RESTAURANT RATING PREDICTION

# # load csv file

# In[143]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('nBagg')
import os
os.chdir('F:\internship')
import pandas as pd
import logging


# In[144]:


logging.basicConfig(filename='model.log',level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger()


# In[142]:


logger.info("dataset is being loaded")


# In[137]:


A=pd.read_csv('zomato.csv')
data=pd.DataFrame(A)
data.head(20)


# # Data Preprocessing and EDA

# as we no need of adrdress,phone,url column because it indicates no value or gives same information for modelling we look for location only

# In[139]:


logger.info("EDA and preprocessing techniques are ")


# In[4]:


data.shape


# In[145]:


drop_col=['url','phone','address', 'listed_in(city)']
data.drop(drop_col,axis=1,inplace=True)


# In[7]:


data.duplicated().sum()


# In[8]:


data.drop_duplicates(inplace=True)


# In[9]:


data.duplicated().sum()


# In[8]:


data.index


# Changing columns name as per our convinence

# In[10]:


data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'listed_type'})
data


# In[11]:


data.votes.describe()


# there are some restaurent with 0 votes too good insights is it!
# and there is a restaurent with 16832 votes too

# In[12]:


print(data.shape)
data.isnull().sum()


# In[13]:


data['rate'].dtype


# In[14]:


data['rate'].unique()


# In[15]:


data['rate']=data['rate'].replace('NEW',np.NaN)
data['rate']=data['rate'].replace('-',np.NaN)
data['rate'].unique()


# In[16]:


data.rate=data.rate.astype(str)
data.rate=data.rate.apply(lambda x : x.replace('/5',''))
data.rate=data.rate.astype(float)


# we can see the rate column is in str type which is non supported type for ML training so we change into float as per the rating method  and need to convert all catogorical column into numerical we will do at time of modelling after visualization for insights

#  so we replaced incorrect value and symbols in rate with NaN value using Replace method

# In[17]:


data['rate'].dtype


# In[18]:


data['cuisines'].isna().sum()


# In[19]:


data.rest_type.value_counts()


# In[20]:


data['rest_type'].isnull().sum()


# In[21]:


data['rest_type'].fillna(value="Quick Bites",inplace=True)


# replacing missing values with most occuring value  in rest_type

# In[22]:


data['cuisines'].value_counts()


# In[23]:


data['cuisines'].fillna('North Indian'or 'North Indian ,Chinese' or 'South Indian' ,inplace =True)


# replacing missing values with most occuring value  in cuisines

# In[24]:


data.isna().sum()


# lets reduce the null values of remaining features using replacing means of others or dropping minimal values

# In[25]:


data['reviews_list'].dtype


# In[26]:


data.reviews_list.values[:1]


# In[33]:


#we could extract these values from reviews and take their mean to fill rate column
data.reviews_list.values[1]


# In[27]:


type(data.reviews_list[1])


# In[28]:


import ast
ast.literal_eval(data.reviews_list.values[1])


# In[29]:


get_ipython().run_line_magic('time', 'data.reviews_list=data.reviews_list.apply(lambda x: ast.literal_eval(x))')


# In[30]:


type(data.reviews_list[100])


# In[31]:


data.reviews_list[0][0][0].split()[0]


# In[32]:


data['reviews_list'].values[1]


# In[33]:


def extract_from_review_list(x):
    #extract the rate value out of a string inside tuple
    # ensure that x is not Null and there is more than one rate
    if not x or len(x) <= 1:
        return None
    rate_new= [float(i[0].replace('Rated','').strip())  for i in x if type(i[0])== str]
    return round((sum(rate_new)/len(rate_new)),1)


# In[34]:


get_ipython().run_line_magic('time', "data['rate_new']=data.reviews_list.apply(lambda x: extract_from_review_list(x))")


# In[35]:


data.loc[:,['rate','rate_new']].sample(10)


# In[36]:


# apply the changes
nan_index = data.query('rate != rate & rate_new == rate_new').index
for i in nan_index:
    data.loc[i,'rate'] = data.loc[i,'rate_new']


# In[37]:


nan_index


# In[38]:


data.rate.isna().sum()


# In[39]:


data.dropna(subset=['rate', 'cost'],inplace=True)


# In[40]:


data.drop(['rate_new'],axis=1,inplace=True)


# In[41]:


data.isna().sum()


# we finllay reduced to zero of null values mostly column 'rate' is filled with mean of review_list for null values 

# 1.reduce the biggest null values of dish_liked with extracting foods stated by reviewrs in reviews_last because its nearly half of the rows filled with null if we remove it is hard for good prediction \
# 2.otherwise fill with most occuring values but it will become bias where null values is huge

# In[42]:


data.dish_liked


# In[43]:


#converting to lowercase
data.dish_liked=data.dish_liked.apply(lambda x:x.lower().strip() if isinstance(x,str) else x)


# In[44]:


data.dish_liked.head(10)


# as we can see dishes liked or disliked are mentioned in reviews so if we can extract these dishes we can fill the nan values of dish_liked column
# we will start by getting a list of all the dishes available from our dataset

# In[45]:


dish_list=[]
for i in list(data.index):
        #print(type(data.dish_liked[i])) #checking whether it shows  index and dish_name type
        if data.dish_liked[i]!='NaN' and isinstance(data.dish_liked[i],str):
            k=data['dish_liked'][i].split(',')
            dish_list.extend(k)
print(dish_list)


# In[46]:


len(dish_list)


# In[47]:


dish_list=set(dish_list) #getting unique dishes values


# In[48]:


len(dish_list)


# In[49]:


p=data.reviews_list[0]
' '.join([i[1].replace('RATED\n ','') for i in p]).replace('\n','').replace('\S+','').replace('?','').replace('Ã','').replace('\\x','').strip().lower()


# In[50]:


# clear the text
def clear_text(t):
    '''
    clear the input text t
    '''
    return ' '.join([i[1].replace("RATED\n  ",'') for i in t]).encode('utf8').decode('ascii',errors='replace'). replace('?','').replace('�','').replace('\n','').replace('.',' ').strip().lower()


# In[51]:


data['reviews_text'] = data.reviews_list.apply(lambda x: clear_text(x))


# In[54]:


dish_list.intersection(data['reviews_text'][100].split())


# In[55]:


data['dish_liked_new']=data.reviews_text.apply(lambda x: ', '.join(list(dish_list.intersection(x.split()))))


# In[56]:


data.dish_liked_new.isna().sum()


# In[57]:


# get sample to compare
data.query('dish_liked != dish_liked')[['dish_liked','dish_liked_new']].sample(5,random_state=1)


# In[58]:


nan_index=data.query('dish_liked !=dish_liked & dish_liked_new==dish_liked_new').index


# In[59]:


get_ipython().run_cell_magic('time', '', "for i in nan_index:\n    data.loc[i,'dish_liked']=data.loc[i,'dish_liked_new'] ")


# In[60]:


data.drop(['dish_liked_new','reviews_text'],axis=1,inplace=True)


# In[61]:


data.isna().sum()


# finally we got them to zero null values let do some binary opertion for simple catogorical values now later we will do encoding

# # simple encoding for simple features

# In[62]:


data['online_order']=pd.get_dummies(data.online_order,drop_first=True)
data['book_table']=pd.get_dummies(data.book_table,drop_first=True)


# In[66]:


data['cost']=data['cost'].astype(str)
data['cost']=data['cost'].apply(lambda v:v.replace(',',''))
data['cost']=data['cost'].astype(float)


# In[67]:


data['cost'].dtype
data['votes'].dtype


# changing type of variables to numerical 

# In[ ]:


logger.info("process is completed")


# In[69]:


logger.info("visulization part of datasets have begin ")


# # INSIGHTS AND VISUALIZATION (using matplotlib,seaborn) OF OUR DATASETS 

# OVERALL VIEW OF OUR DATASETS)

# In[64]:


sns.pairplot(data=data)


# In[65]:


data['book_table'].value_counts()


# In[63]:


sns.countplot(data.book_table,palette = "Set1")
plt.title('No of tables booked and not booked  0=No 1=Yes ')
plt.show


# In[64]:


data['online_order'].value_counts()


# In[65]:


sns.countplot(data.online_order,palette = "dark:salmon_r")
plt.title('count of Online order  0=No 1=Yes ')
plt.show


# RESTAURANTS NAME AND ITS COUNTS 

# In[68]:


data['name'].value_counts()


# In[66]:


plt.figure(figsize=(10,5))
data.name.value_counts()[:10].plot(kind='bar',color='black')
plt.title("Top 10 Restaurants and Names",weight='bold')
plt.ylabel('count')
plt.show()


# INSIGHTS OF LOCATIONS AND RESTAURANTS 

# In[70]:


data['location'].value_counts()


# In[71]:


plt.figure(figsize=(15,15))
ax =data.location.value_counts()[:15].plot(kind='bar',color='c')
plt.title('Number of Restaurants in each location top 15', weight='bold',)
plt.xlabel('location')
plt.ylabel('No. of Restaurants')
plt.show()


# In[72]:


plt.figure(figsize=(12,7))
colors = ['gold', 'red', 'lightcoral', 'lightskyblue','blue','green','silver','violet']
explode = (0.4, 0, 0, 0,0,0,0,0,0,0)  # explode 1st slice
ax=data.location.value_counts()[:10].plot(kind='pie',explode=explode,shadow=True,colors=colors,startangle=140,autopct='%1.1f%%')
plt.axis('equal')
plt.title("Percentage of restaurants present in that location", weight = 'bold')


# RESTAURANTS TYPE AND COUNTS 

# In[73]:


data['rest_type'].value_counts()


# In[74]:


plt.figure(figsize=(12,9))
colors = ['purple','brown', 'red', 'lightcoral', 'lightskyblue','blue','green','silver','pink','yellow']
ax=data.rest_type.value_counts()[:10].plot(kind='pie',colors=colors,startangle=180,autopct='%1.1f%%')
plt.axis('equal')
plt.title("Percentage of restaurants_type", weight = 'bold')


# In[67]:


M=data['cost'].max()
m=data['cost'].min()
a=data['cost'].mean()
print('maxiumum cost is {0} , mimimum cost is {1} , average cost per two is {2}'.format((M),(m),(a)))


# Listed_type is visualized using treemap for better visual of counts and understadings

# In[68]:


plt.figure(figsize=(12,7))
data.listed_type.value_counts().plot(kind='bar')
plt.show()


# In[227]:


data['listed_type'].value_counts()


# In[77]:


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


# In[79]:


plt.figure(figsize=(5,5))
sns.distplot(data['cost'],color='blue')
plt.show()


# lets check Normal distribution of our label(y)

# In[80]:


data.rate.hist(color='b',bins=30)
plt.axvline(x=data.rate.mean(),color='red',ls='--')
plt.title("Restaurant's ratings")
plt.xlabel('Rating')
plt.ylabel('No. of Restaurants')
plt.show()
print(data.rate.mean())


# Lets go into little good Insights

# In[81]:


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


# In[ ]:


logger.info("visulaization is occured with good insights")


# # Feature Engineering ( Label Encoding) 

# LABEL ENCODING AND DROP UNREQUIRED COLUMNS FOR TRAINING

# In[ ]:


logger.info("Feature engineering occured for changing catogorical into numerical")


# In[69]:


fdata=data.copy()


# In[71]:


fdata.drop(['reviews_list','name'],axis=1,inplace=True)
fdata.drop(['dish_liked'],axis=1,inplace=True)
fdata.drop(['listed_type'],axis=1,inplace=True)


# In[72]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
list1=['rest_type','location','cuisines','menu_item']
for i in list1:
    fdata[i]=LE.fit_transform(fdata[i])
fdata.head()


# In[73]:


fdata.sample(10)


# # Checking multicolinearity  and correlation factors 

# In[146]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

X=fdata[['online_order','book_table','votes','location','rest_type','cuisines','cost','menu_item']]

vif= pd.DataFrame()
vif["features"] = X.columns
vif['VIF']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)


# as VIF is below 10 so it is normal to have the getted VIF values for multicolinearity

# # Data transformation and Features Scaling

# WE USING LABEL ENCODED FEATURES FOR OUR MODEL BUILDING

# In[75]:


fdata.rate.unique()


# In[76]:


fdata.shape #using label encoder


# In[77]:


Features=fdata.drop(['rate'],axis=1)
Features.shape


# In[78]:


Features.dtypes


# In[148]:


corr=Features.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)


# In[69]:


fdata['rate'].unique()


# In[79]:


label=fdata['rate'].values
label


# # Splitting Datasets

# Splitting the datasets into train and test data to see how our model performs to untrained datas
# train set = 80% and test set =20%

# In[140]:


logger.info("splitting datasets into train and test data to perform modeling")


# In[81]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(Features,label,test_size=0.2,random_state=42)


# In[82]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# # REGRESSION MODELS

# # 1.linear regression

# In[114]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
logger.info('linear regression is being called for modeling')


# In[106]:


LR.fit(X_train,y_train)
logger.info("data is trained")


# In[107]:


print(LR.score(X_train,y_train))
print(LR.score(X_test,y_test))
print('intercept =',LR.intercept_)
print('coefficient =',LR.coef_)
logger.info("scores of test data and train data are obtained")


# PREDICTION USING Linear Regression

# In[111]:


YP=LR.predict(X_test)
logger.info('label is predicted using features as {}'.format(YP))


# In[99]:


from sklearn.metrics import r2_score
LR=r2_score(y_test,YP)*100
print("Accuracy score for LR:",LR)
logger.info('accuracy of linear regression is {}'.format(LR))


# Prediction is not up to the mark in linear regression as it score is just 25% 

# # 2. Random Forest Regressor

# In[116]:


from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(n_estimators=650,random_state=245,min_samples_leaf=.0001)
RF.fit(X_train,y_train)
RF_predict=RF.predict(X_test)
logger.info('Random Forest Regressor is being called for modeling')
logger.info(RF_predict)


# In[77]:


print(RF.score(X_train,y_train))
print(RF.score(X_test,y_test))


# SAMPLE TESTING

# In[274]:


#new=pd.DataFrame({'a':[1],'b':[0], 'c':[918],'d':[1],'e':[20],'f':[662],'g':[800.0],'h':[8519]})
#predicted=RF.predict(new)
#predicted


# lets compare our obtained output with actual output

# In[89]:


sample=pd.DataFrame({"Actual Rating":y_test,
             "Predicted Rating":np.round(RF_predict,2)})
sample


# R^2 = (Y-Y^)^2 ------ (expected-predicted)^2

# In[117]:


from sklearn.metrics import r2_score
RFr2=r2_score(y_test,RF_predict)*100
print("Accuracy score for RF:",RFr2)
logger.info('Accuracy of Random Forest is {}'.format(RFr2))


# # 3.Support Vector Machine

# In[118]:


from sklearn.svm import SVR
svr=SVR()
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',kernel='rbf', max_iter=-1, 
    shrinking=True, tol=0.001, verbose=False)
svr.fit(X_train,y_train)
print(svr.score(X_train,y_train)*100)
print(svr.score(X_test,y_test)*100)
logger.info("support vector is being called for modeling")


# In[120]:


svr_pred=svr.predict(X_test)
svr_pred


# In[121]:


svr=r2_score(y_test,svr_pred)*100
print("Accuracy score for RF:",svr)
logger.info('Accuracy of Support vector Machine is {}'.format(svr))


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


# In[ ]:


br=r2_score(y_test,bayesin)*100
print("Accuracy score for RF:",br)
logger.info("BayesinRidge is called for accuracy and accuracy is {}".format(br))


# # ExtraTree Regressor

# In[123]:


from sklearn.ensemble import  ExtraTreesRegressor
ETR=ExtraTreesRegressor(n_estimators = 150)
ETR.fit(X_train,y_train)
y_pd=ETR.predict(X_test)
print(y_pd)
logger.info('Extra Tree Regressor is being called ')


# In[79]:


ETR.score(X_test,y_test)


# In[81]:


ETR


# In[124]:


from sklearn.metrics import r2_score
EXR=r2_score(y_test,y_pd)*100
print("Accuracy score for RF:",EXR)
logger.info('Accuracy of Extra Tree Regressor is {}'.format(EXR))


# ET Regressor is giving best than random forest as it is like same as random forest but it use whole training data set for trees and 
# uses randomly split unlike random forest of optimal split and its an ensemble technique bootstrap(bagging)

# In[95]:


sample=pd.DataFrame({"Actual Rating":y_test,
             "Predicted Rating":np.round(y_pd,2)})
sample.sample(5)


# In[131]:


from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pd)))
logger.info('rmse score is {}'.format(rmse))


# # Ridge Regression

# In[133]:


from sklearn.linear_model import Ridge
rr=Ridge()
rr.fit(X_train,y_train)
print(rr.score(X_train,y_train))
print(rr.score(X_test,y_test))
rr_pred=rr.predict(X_test)
rr_score=r2_score(y_test,rr_pred)*100
print("Accuracy score for RidgeR :",rr_score)
logger.info('ridge regression is called and accuracy is {}'.format(rr_score))


# # DATA MODELS ACCURACIES

# In[361]:


Final_scores=pd.DataFrame({"Model Names":['Linear Reg','RandomForest Reg','Support Vector Reg','Bayesin Ridge','Extra tree Reg','Ridge Reg'],
            "Accuracy socre":[LR,RFr2,svr,br,EXR,rr_score]})
Final_scores


# # Saving Models 

# Saving best model in pickle to use for further prediction

# In[132]:


import pickle as pkl
import _pickle as cPickle
import bz2


# In[86]:


#with open("ETRmodel",'wb') as file:
 #   pkl.dump(ETR,file)


# In[364]:


Final_scores.to_csv("Models Prediction Scores.csv",index=False)


# PICKLE FILE IS TOO LARGE SO USED BZ2 file for compressing

# In[ ]:


zipfile=bz2.BZ2File('model','wb')
cPickle.dump(ETR,zipfile)
logger.info("model was dumped in pickle file using bz2 because of large file")


# In[107]:


fd=bz2.BZ2File('model','rb')
etr=cPickle.load(fd)
logger.info("model was loaded from pickle for predicting")


# In[108]:


etr.predict(X_test)


# PERFORM WITH DIFFERENT PROTOCOL TO MINIMIZE MEMORY

# In[123]:


#n=bz2.BZ2File('pickle','wb')
#cPickle.dump(ETR,n, protocol=3)


# In[124]:


#nm=bz2.BZ2File('pickle','rb')
#r=cPickle.load(nm)


# In[125]:


#r.predict(X_test)


# # Conclusion 

# From the analysis, 'Onesta', 'Empire Restaurant' & 'KFC' are the most famous restaurants in bangalore.
# Most Restaurants offer options for online order and delivery. Most restaurants don't offer table booking. 
# From the analysis, most of the ratings are within 3.5 and 4.5.
# From the analysis. we can see that most of the restaurants located in 'Koramangala 5th Block', 'BTM' & 'Indiranagar'.Then least restaurants are located 'KR Puram', 'Kanakapura', 'Magadi Road'.
# 
# 'Casual Dining', 'Quick Bites', 'Cafe', 'Dessert Parlor' are the most common types of restaurant. And 'Food Court', 'Casual Dining', 'Dhaba' are the least common.
# 
# From the analysis, pasta & Pizza most famous food in bangalore restaurants. From the analysis, we can see that North Indian Cuisines are most famous in bangalore restaurants. Two main service types are Delivery and Dine-out. From the analysis, we can see that 'Onesta', 'Truffles' & 'Empire Restaurant' are highly voted restaurants.
# 
# For the modeling part, I used LinearRegression, DecisionTree Regressor, RandomForest Regressor , Supprotvector Regressor & ExtraTree Regressor. From all these models ExtraTree Regressor perform well compared to the other models.So i selected ExtraTree Regressor for model creation
