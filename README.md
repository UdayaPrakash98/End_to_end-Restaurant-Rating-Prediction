# Zomato Restaurant-Rating-Prediction
![zomato](https://user-images.githubusercontent.com/81813682/129445907-1e1976d1-66c1-4f5b-961b-de9ba1578176.jpg)

## Project Overview:
<p>Restaurants from all over the world can be found here in Bengaluru. 
From United States to Japan, Russia to Antarctica, you get all type of cuisines here. 
Delivery, Dine-out, Pubs, Bars, Drinks, Buffet, Desserts you name it and Bengaluru has it.
Bengaluru is best place for foodies. The number of restaurant is increasing day by day.
Currently which stands at approximately 12,000 restaurants. With such a high number of restaurants. 
This industry hasn’t been saturated yet. And new restaurants are opening every day. 
However, it has become difficult for them to compete with already established restaurants. 
The key issues that continue to pose a challenge to them include high real estate costs, rising food costs, shortage of quality manpower, fragmented supply chain and over-licensing. 
This Zomato data aims at analyzing the demography of the location. 
Most importantly it will help new restaurants in deciding their theme, menus, cuisine, cost etc for a particular location. 
It also aims at finding similarities between neighborhoods of Bengaluru on the basis of food.</p>

## PROBLEM STATEMENT:

<p>The main goal of this project is to perform extensive Exploratory Data Analysis(EDA) on
the Zomato Dataset and build an appropriate Machine Learning Model that will help
various Zomato Restaurants to predict their respective Ratings based on certain
features.</p>

## METHODS FOR PROCESSING DATA
<p>The main goal is to predict Zomato Restaurant Rating on various featuresavailable in the dataset.</p> 
<p>The classical machine learning tasks like Data Exploration, Data Cleaning,
Feature Engineering, Model Building and Model Testing. Try out different machine
learning algorithms that’s best fit for the above case.</p> <pre>
<li> Data Exploration     : I started exploring dataset using pandas,numpy,matplotlib and seaborn. </li>
<li> Data visualization   : Ploted graphs and bars to get insights about dependend and independed variables. </li>
<li> Feature Engineering  : 1.Removed missing values and created new features as per insights.
                           perform label encoding to change categorical variables into numerical ones.</li>
<li> Model Selection I    : 1.Execution of  models to check the base accuracy.
                           Also find residual and R2 to check whether a model is a good fit or not.</li>
<li> Pickle File          : Selected model as per best accuracy and created pickle file .</li>
<li> Webpage & deployment : Created a Flask application that takes all the necessary inputs from user and shows output.
                            After that I have deployed project on heroku and AWS service</li></pre>
																

## Technologies used in project
<pre> 
1. Python 
2. Sklearn
3. Flask
4. Html
5. Css
6. Pandas, Numpy
7. Matplotlib,seaborn
8. Database
</pre>

## Application Interface:

![Screenshot (91)](https://user-images.githubusercontent.com/81813682/129447522-7941694a-079d-4738-99f9-1e13902fb44e.png)


## Deployment Links:
<p> Link Heroku : <a href="https://rating-prediction.herokuapp.com/" <br>
App Service Application URL:  </p>

</pre>

## High Level Design Document :

## Low Level Desgin Document :

## Quick tip for prefer EDA : 
<p>just use for fast EDA process to view your datasets dont dependnt on it </p>
<p>import pandas_profiling</p>
<p>pandas_profiling.ProfileReport(data)</p>

