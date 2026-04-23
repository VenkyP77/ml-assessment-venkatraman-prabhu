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
I will do the following four analyses (charts/metrics):
1. **Monthly sales distribution (histogram)**
    - I will look for skew, heavy tails and outliers (months that are very different from the rest) in the data.
    - **Influence on my modelling approach:** If the data is skewed or presents a long-tail distribution, I may need to consider logarithmic transformations to the dataset to do some pre-proccessing for stores for which results may be hard to predict.
2. **Calendar and seasonal effects (time series by month)**
    - Here I will look for peaks that happen again and again (like festivals or the end of a season), or structural breaks that may indicate marketing policy changes.
    - **Influence on my modelling approach:** I will need to use **time-based validation** (walk-forward), and not do random splits that let future seasonality into training. Temporal splits would be the way to go.
3. 
