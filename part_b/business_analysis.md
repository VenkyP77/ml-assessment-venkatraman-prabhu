# Part B - Business Case Analysis

## The Context
A fashion store has 50 locations in cities, suburbs, and rural areas. There are five deals that are used every month - "Flat Discount", "BOGO", "Free Gift with Purchase", "Category-Specific Offer", and "Loyalty Points Bonus". Stores vary in size, how many people go there, how much competition they have, and who shops there and how much. The goal is to choose which promotion to use in each store every month so that they can sell the most items.

## B1. Problem Formulation
**
### a. Machine Learning Problem Statement
1. The Target variable (Y) is the total number of items sold (sum of quantities of each item), in a defined time-window which determines how promotions are decided. The best way to define this is - **total number of items sold (Y)** for the store (s) in a month (m) when a promotion (p) was active
2. The candidate input features are derived from ***store context*** [location type, store size, estimated monthly footfall, local competition density, customer demographics] and **promotion details** joined to the store-month assignment as a categorical encoding of the 5 promotion types
3. This is an example of **supervised learning** machine learning problem, in which the learned function predicts how many items will be sold under each possible promotion for a given store and month. I recommend using **regression** with **promotion type as a categorical input feature** to find **five predicted volumes** (one for each promotion) for each candidate store-month. We can then choose the promotion with the highest predicted volume.
**REASONING:** In this case, we have a clear target to meet. Since, in this case we want to calculate a total item count, we need a model that can be measured by how close its guess is to the actual historical count. Also, since our outcome is a quantity, this falls under a sub-category of Supervised Learning called Regression. This model will look at all the different promotions we have run in the past and calculate the "weight" or impact of each one. Once it is trained, we can give it a future scenario to provide a prediction.

### b. Why sales volume is a better measure for this goal?
Revenue is the combination of price and quantity. Promotions change **prices**, **effective price per item** (discounts), and **bundle composition** (BOGO, gifts) directly. While two promotions can make the same amount of money, each may achieve that by selling very different amounts of units. This means that revenue is not just based on how many customers buy something, but also on how prices work. For example, BOGO and free gifts change how revenue is recorded based on how many units are sold (bundling, multi-SKU effects). Hence, using "items sold" is more in line with the stated goal of "maximizing sales volumes", regardless of how easy it is to add up revenue for accounting purposes. Another important factor is that when interventions also change prices and the structure of the promotion, volume becomes a more comparable outcome across different types of offers irrespective of whether they are structurally similar or different.

### c. Better Alternative to a Single Global Model across all stores
There may be differences in baseline demand, seasonal changes, and how stores respond to the same promotion scheme between stores in cities and stores in the country. A single global model might "average away" the differences. This will make decisions worse, or the model may "memorize" the store ID if the store is just one-hot encoded and has no clear structure. So, instead of using one global model that is equally sensitive to promotions everywhere, we should think about using a model that will clearly allow for different baselines and different promotion sensitivities by store or by market.
**Proposed modelling approach: ** We can group stores into clusters based on their behavior (demand patterns, demographics, or foot traffic) instead of having one model for everything or one model for each store. We train a shared backbone that will learn general demand drivers (like how price drops boost sales), but the last layers of the network will be more specific to the cluster, like Urban, Rural, Semi-Urban, etc. This method should stop the "averaging away" problem because the rural head only learns from rural data, but it can still use the "Global" backbone's knowledge of how promotions work.

## B2. Data and EDA Strategy
### a. Raw Data preparation
1. The "Transactions" table has at least "store_id," "transaction datetime/date," "line-level quantity," and "sku," "promotion_id," or flags that link with promotions details.
2. We can use **store_id** to link **store_attributes** to transactions (each store has a lot of transactions).
3. We should also add weekend and festival indicators (and any other calendar fields) by joining **calendar** on **date**. We can also shorten the transaction timestamp to the calendar date.
4. We can use the **promotion_id** from transactions to join **promotion_details** if promotions are given out at that level instead of being guessed from the transactions.
5. We can have one row per **store_id x month**. This way there is one row for each **Result**. Hence, the total number of items sold in a particular store-month, would be **items_sold_month**.
6. Before starting modelling we should also define following aggregations: 
    - **sum(quantity)** as the goal 
    - **count(distinct transaction_id)**
    - **count(distinct sku)**
    - **count(festival_days)**

### b. EDA before modeling
I will do the following four analyses (charts):
1. **Monthly sales distribution (histogram)**
    - I will look for skew, heavy tails and outliers (months that are very different from the rest) in the data.
    - **Influence on my modelling approach:** If the data is skewed or presents a long-tail distribution, I may need to consider logarithmic transformations to the dataset to do some pre-proccessing for stores for which results may be hard to predict.
2. **Calendar and seasonal effects (time series by month)**
    - Here I will look for peaks that happen again and again (like festivals or the end of a season), or structural breaks that may indicate marketing policy changes.
    - **Influence on my modelling approach:** I will need to use **time-based validation** (walk-forward), and not do random splits that let future seasonality into training. Temporal splits would be the way to go.
3. **Correlation heatmap of numeric store attributes and baseline monthly volume.** 
    - I need to look for multicollinearity between parameters (for example, footfall vs. size)
    - **Influence on my modelling approach:** If collinearity is observed, then I may need to drop one of the correlated variables if they measure the same effect, OR look at feature transformation (e.g. use a ratio of the correlated variables - footfall by size). I can also consider using regularization approaches with linear modelling or go with Gradient Boosting / Random Forest models.
4. **Bar chart for Average volume by promotion AND stacked bar of share of months each promotion runs**
    - I will look for an imbalance (e.g. rare promotions), overlap with seasons, and/or clear patterns of non-random assignment. For instance, if a promo shows high average volume, but on the stacked bar I can see that it only runs in October, the increased volume may be due to the Diwali festival, rather than the promo itself. Conversely if the share of months for a specific promo is very small, the "Average Volume" statistic become unreliable.
    - **Influence on my modelling approach:** I will need to introduce recency / frequency indicative features instead of a binary feature e.g. Instead of just capturing "If Promo is Active", create a feature like "Days since the last promo run". Also including interaction features like "Month" X "Promo Type" can potentially mitigate the effect of seasonality.

### c. Dataset imbalance and its effect on the model
When 80% of the data represents the baseline (no promo), the model can discover that it can achieve 80% accuracy simply by predicting the baseline every time. This could have a significant impact on the ability of the model to accurately predict the outcome.
Mean Squared Error (MSE) and other standard loss functions are heavily influenced by big outliers. The model will focus on getting the non-promotional periods right because they are the "standard." However, when a promotion happens with a corresponding spike in volume, the model will undershoot the prediction because the weight of the 80% baseline will pull the prediction toward the mean.
To address this, I can use a pipeline with 2 separate models, instead of a single model. One model can be a "Classifier" model which predicts if there was a significant shift from the baseline. The second model which I will train only on data from promotional periods can then be used to predict the magnitude of the impact of the promotion. I will then also track the errors separately - "Baseline error" for non-promo periods and "Promo Error" for specifically promo periods.


## B3. Model Evaluation and Deployment

### a. Evaluation Metrics
In this case, where we have monthly store data spanning 3 years across 50 stores, there is huge time dependency on the data. For example what happened at any particular store in any month could be highly dependent on what happened in that store in the preceding month or several months. If we randomly sample the data and do the train-test split, we might end up with a scenario where we train the model on future months and test it on past months. This is not right, since logically it means that the model looks at the future data to predict the past, instead of the other way around.

**Proposed approach:** In this case, I will use a time-series split of the data. For example, I will train the model on the first 24 months of the dataset and test it on the remainder 12 months.

**Evaluation Metrics and Interpretations:** Since the goal is to maximize items sold (a continuous count), I will treat this as a regression problem to predict sales under each of the five promotion types. Then I can select the one with the highest predicted value.
I will use **Mean Absolute Error (MAE)** which can indicate, on an average, how many items our prediction is "off" by. For example if the MAE is 10 for average sales of 100 by a store, I know that the promotion planning has a 10% margin for error.
Using the **Mean Absolute Percentage Error (MAPE)** can help me by normalizing the error as a percentage. This will help because the absolute error needs to be seen in the context of the overall volume. For instance, an error of 100 items for a Urban store with high sales volumes may be acceptable, but for a small rural store with low volumes, this might turn out to be a disastrous prediction. 
I can also look at **Root Means Squared Error (RMSE)**. Since RMSE penalizes larger errors more heavily as compared to MAE, in conjunction with the MAE, it can help me identify if the model is failing on predicting outliers. This can help me ensure that my predictions DO NOT lead to massive stockouts or losses. For example, if my model is predicting over-predicting by a large margin for a store, I could end up having to markdown prices due to lack of sales. In such cases, a high RMSE vs MAE can clearly indicate that my model is over-predicting.

### b. Communicating model recommendations
After training, the model recommends the Loyalty Points Bonus for Store 12 in December and the Flat Discount for Store 12 in March. 
To investigate this, I will first isolate the observations for store 12 in December and March and look at the model's internal decision logic. I can then identify which features contributed to the prediction of "Loyalty Points Bonus" in December and "Flat Discount" in March. I will also look at specific variables that changed significantly between the 2 months. This analysis may give me insight like, December was a high-footfall month because of the holiday season and March was a low-footfall month. This could be the reason that the model recommended a "Loyalty Points Bonus" in December to ensure that repeat visits happen in January which could potentially be lower-footfall and it might increase the basket size without cutting into immediate profit margins. In case of March, there could be higher competition density with shoppers being less motivated and nearby stores running Promotions. In this case a "Flat Discount" is intended to hook the customers into coming into the store. 
In order to communicate and explain the model predictions to the Marketing team, I will focus on the following:
1. I will ensure that I avoid all technical jargon - "feature contributions", "weights", "coefficients", etc. and focus on communicating the business takeaways
2. I will explain how the various factors and their historical data like "store footfall", "sales volumes", etc. are considered by the model during its training, to make the predictions
3. I will explain the primary goal of the model while making the prediction. For example - In December, the primary goal for the model was to "Retain and Reward" customers, whereas in March it was to "Acquire and Increase traffic". I will also highlight the key variables that led the model to the conclusion and the insight the model could provide. For example in December the insight was that Customers were already coming into the store because of the festive season. So it made sense to ensure that spends were maximised and by using loyalty points get them to come back in January which could be potentially lower footfall.
4. I will also visualize and demonstrate the potential impact of the promotions on baseline sales, to arrive at the model predicted values.
5. Finally I will ensure that the marketing team realizes that the predictions made by the model are NOT "one-size-fits-all". I can get this validation from the marketing team by cross-checking some of the insight from the model with real-world insight from the marketing team. For instance, a dip in foot-fall in March last year could have been because of some local event near the store which disrupted store operations.

### c. 
