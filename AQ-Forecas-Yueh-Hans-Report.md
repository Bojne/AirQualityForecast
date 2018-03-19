# AQ Forecas - Yueh Han’s Report

> work by Yueh Han Huang, for KKBOX DS intern test, work period: 3.17 - 3.18


## Task

Please use the public data set in the above link to build a model to predict air quality in the upcoming days. You may use your own definition for “Air quality.” It may depend solely on hourly CO concentration for the past 30 days. It may depend on Benzene concentration for the past 7 days. It can also depend on the concentration of a mix of chemicals of your choice.
Given your definition of air quality, please predict air quality for the next 5 days.

## Points to note
- We are NOT asking for an accurate prediction
- We are more interests in your thought process and your methods to tackle the problem
- Some of the areas we are looking for:
  - “YOUR” definition of air quality and the reasoning behind it
  - The logic for picking the model of your choice
  - How you evaluate the causal relationship between different chemical concentrations as well as the relationship between chemicals and air quality
  - The code used for your analysis/prediction
  - Model validation methods
  - Visualization of your data and result
- DEFINITE FAIL
  - Trying out textbook model (e.g. SVM, RNN, ARIMA, etc) WITHOUT DETAILED interpretation of results.
- With limited time, we know it is hard to complete everything perfectly. It is good enough to demonstrate the areas you are most comfortable with. 
- Good luck
----------
## Notes 
- Outline is generated 
# **Abstract**


# 0/**About** **Dataset**

[data source](https://archive.ics.uci.edu/ml/datasets/Air+Quality#) 

## Data Information summary
- Size The dataset contains 9358 instances . 
- Features 10 chemicals + 2 time_related + 1 temp + 2 humidity = 15 features
- Location The device was located on the field in a significantly polluted area, at road level, within an Italian city.
- Period Data were recorded from March 2004 to February 2005 (one year) 
- Indicators Ground Truth hourly averaged concentrations for CO, Non-Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) 
- Missing Missing values are tagged with -200 value.
## Related Data Sets? Can’t find

It would be great to have windspeed and wind direction data associate with our data. However it’s 
unclear where is the exactly location of the data, so there’s no way to consider related data sets 
I gauss it would be also be helpful too add forecasting temperature into our test set to predict the AQI. But again it can’t be accessed. 

# 1/**Data Pre-Processing**

Entire Process and be read on My Python Code
****
## Understanding / Formatting  / Cleaning 
| **Problems**                                                  | **Solutions**                | **Related Code**                                                   |
| ------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| Totally Empty columns                                         | Drop them                    | `df.drop(['Unnamed: 15', 'Unnamed: 16'])`                          |
| Totally Empty rows                                            | Drop them                    | `df = df.dropna(how = 'all')`                                      |
| Float columns has string value. 
( Cause: use , instead of .) | Replace and change data type | `df[col].str.replace(',','.')` 
`df [col] = df[col].astype(float)` |
| columns order                                                 | reindex columns              |                                                                    |

## Feature engineering 
1. Generated Air Quality Index(AQI), explained in the next sections 
2. From date time to generated the following features
  - Date(2004/05/10),  Day(10),  Week(20),  Weekday(1),  Month(5),  Hour(15)
  - Opinion *This procedure is important for periodic data (e.g. Air Quality, Electricity Usage), from them I think Weekday, Week, Hour is the most important indicators in dt-type features.*
## Missing values

**Problems**
1/ Missing values percentage for each AQ features, from the  

    # from def missing_value_pct(df_name)
    PT08.S1(CO) :  4 %
    PT08.S2(NMHC) :  4 %
    PT08.S3(NOx) :  4 %
    PT08.S4(NO2) :  4 %
    PT08.S5(O3) :  4 %
    CO(GT) :  18 %
    C6H6(GT) :  4 %
    NOx(GT) :  18 %
    NO2(GT) :  18 %
    T :  4 %
    RH :  4 %
    AH :  4 %

2/ Missing values distribution 
Take CO(GT, 18 %) with 1683 missing values as example. 
*In weekday, Month, the missing values distribution seems normal, in hour 4 am there’re most missing value.* 

![Missing values counts in weekday](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521428404754_p3.png)
![Missing values counts in month](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521428404811_p2.png)
![Missing values counts in hour](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521428404846_p.png)

![Missing values counts in weekday](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521428772122_image.png)


**Solutions: fill with previous data point**
3/There are several ways to dealing with missing values in this case:

  1. Fill na with previous data point
  2. Intermediate methods, like exponentially average or take last week-day-hour data to implement
  3. advance methods like k-NN algorithm
  
  *Due to the limited amount of time, I’ll just take the easy&fast way: fill missing values with previous value. Since for most features there’s only 4% missing, and the missing distribution is not obviously on specific days, the fill method should not have huge impact.*
# **2/Air Quality Index | Definition and Reasoning**

Defining Air Quality Index is an art, for few reasons:

- for different application we needs different indicators(e.g. )
- we have to consider the patterns of the data.
- the background knowledge about the chemical is essential. 

Simply put, I choose the concentration of PT08.S4 to make AQ index for NOx. I used the exponentially weight moving average (ewma) to generate AQI. The method is explain in the followings:

## **Why** **PT08.S4(NO2)** **as indicator?**
![NO2 → 03](http://aqicn.org/images/nowcast/ozone-as-a-secondary-pollutant.png)

- 二 氧 化 氮（ NO2 ） 生 成 原 因 係 來 自 燃 燒 過 程 中 ， 空 氣 中 氮 或 燃 料 中 氮 化 物 氧 化 而 成 
- 二 氧 化 氮 為 具 刺 激 味 道 之 赤 褐 色 氣 體 ， 易 溶 於 水 ， 與 水 反 應 為 亞 硝 酸 及 硝 酸 ， 在 空 氣 中 可 氧 化 成 硝 酸 鹽 ， 亦 是 造 成 雨 水 酸 化 原 因 之 一 。
- Data only 4% missing
- std is relatively small(compare to O3)
- correlated to other data
- Is the primary air pollutant, produce other secondary pollutants such as O3
## **Hourly AQI**

Define by Exponentially Weight Moving Average (ewma).  
Center of mass (C) is set as 2. ($$\alpha = \frac{1}{1+C}$$)

## **Daily AQI**

Day define as 00:00 ~ 23:00
For a long scale view, we generate Daily AQI from the mean of $$3$$ highest hourly AQI in the day
Daily AQI = $$\frac{\Sigma_i^3 {G_i}}{3}$$, where $$G_i = top_3$$( **Hourly AQI** ).mean()

## **Behind definition**
## **Code to generate AQI**
    center_of_mass = 2
    # define
    def add_hourly_aqi(df, chemical):
      if chemical not in df.columns:
        print('{} no in dataframe!'.format(chemical))
      else:
        df['ewma'] = df[chemical].ewm(min_periods=0,adjust=True,ignore_na=False,com=center_of_mass).mean()
    add_hourly_aqi(df2,'PT08.S3(NOx)')
# 3/**Data Analysis and Visualization**
## Understand the feature distribution
1. We Can group our features as 3 types : timestamp, AQ_features and Temp+Humd
2. Timestamp: Trivial 
3. AQ_features: I made two sub group, one
![Histogram for every features](/static/img/pixel.gif)

![We can see O3 has highest std](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521448602597_image.png)

****## Periodic Data

Since it’s periodic data, we and make data group by Weekday, Week, Month,  etc to see it’s patterns. 

## **AQI Changes**
- Q1 What’s the change in Weekdays?
- Q2 What’s the change in Weeks?
- Q3 What’s the change in Month?
- Q4 What’s the change in Hours?
  - 4-1 w.r.t. Weekdays (7 legend)
  - 4-2 w.r.t. Season (12 legend)
![Q1 What’s the change in Weekdays?](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451134928_image.png)

![Q2 What’s the change in Weeks?](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521450863588_image.png)

![Q3 What’s the change in Month?](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521449096061_image.png)

![](/static/img/pixel.gif)

![Q4-1 What’s the change in Hours w.r.t. Weekdays (7 legend)?](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521449375624_image.png)

![Q4-2 What’s the change in Hours w.r.t. Seasons (4 legend)?](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521450897427_image.png)

## **For individual Chemicals** 
![Abs Humidity change in hour. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526224_plot+AH+change+in+hours.png)
![PT08.S5(O3) change in our. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526915_plot+PT08.S5O3+change+in+hours.png)

![Average CO(GT) change in hour. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526682_plot+COGT+change+in+hours.png)
![Average NO(GT) change in hour. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526752_plot+NO2GT+change+in+hours.png)

![Average PT08.S1 change in hour. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526873_plot+PT08.S1CO+change+in+hours.png)
![Average PT08.S2 change in hour. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526814_plot+PT08.S2NMHC+change+in+hours.png)

![Average PT08.S4(NO2) change in hour.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526900_plot+PT08.S4NO2+change+in+hours.png)
![Average Relative Humidity change in hour.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526257_plot+RH+change+in+hours.png)

![Average C6H6(GT) change in hour.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526711_plot+C6H6GT+change+in+hours.png)
![Average temparature change in hour.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526601_plot+T+change+in+hours.png)



![Average PT08.S3(NOx) change in hour.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526839_plot+PT08.S3NOx+change+in+hours.png)
![NOx(GT) change in our. Line for different weekday.](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521451526789_plot+NOxGT+change+in+hours.png)


**Observation** 


## **Between Chemicals** 

**Visualizing correlation matrix**

![Correlation Matrix on 15 features](https://d2mxuefqeaa7sj.cloudfront.net/s_42A6F10B5AE9BD5DE26D22AE518A57D6B5EF48F57CAD1A3693978CADCC1A2528_1521452319773_image.png)



## **Chemicals and AQI**
- Relationship betweens chemicals and AQI


    idx             -0.547952
    PT08.S3(NOx)    -0.521679
    Weekday         -0.110188
    RH              -0.042416
    Day             -0.028217
    NO2(GT)          0.108375
    NOx(GT)          0.190435
    Week             0.193906
    Month            0.209171
    Hour             0.223788
    Season           0.397805
    CO(GT)           0.567791
    PT08.S5(O3)      0.573614
    T                0.589130
    AH               0.653837
    PT08.S1(CO)      0.655639
    C6H6(GT)         0.730844
    PT08.S2(NMHC)    0.749167
    PT08.S4(NO2)     0.988260
    AQI              1.000000
# 4-1/**Predicting** 

(Code finished but still working on optimizing and presenting)

## Training Set 
- Split data 
- Model choosing 
- Model selection, validation
# 4-2/**Model-tuning** 

(Code finished but still working on optimizing and presenting)
****
# **Reference** 
## Air Pollution and chemistry
- [Nitrogen Dioxyde (NO2) in our atmosphere](http://aqicn.org/faq/2017-01-10/nitrogen-dioxyde-no2-in-our-atmosphere/)
- [A Beginner's Guide to Air Quality Instant-Cast and Now-Cast.](http://aqicn.org/faq/2015-03-15/air-quality-nowcast-a-beginners-guide/)
- [Air quality index](https://www.wikiwand.com/en/Air_quality_index)  WIKI
- [NowCast (air quality index)](https://www.wikiwand.com/en/NowCast_(air_quality_index)) 

[](https://www.wikiwand.com/en/NowCast_(air_quality_index))
[](http://aqicn.org/faq/2017-01-10/nitrogen-dioxyde-no2-in-our-atmosphere/)


