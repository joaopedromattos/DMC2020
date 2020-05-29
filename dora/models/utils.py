import zipfile
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

DATA_FILE = "1.0v.zip"


def read_data(data_dir="../main/datasets/", data_file=DATA_FILE):
    """Returns the data, in order infos, items, orders"""
    with zipfile.ZipFile(data_dir+DATA_FILE) as z:
        dfs = []
        for name in ["infos", "items", "orders"]:
            dfs.append(pd.read_csv(z.open(f"1.0v/{name}.csv"), sep="|"))
    return dfs


def process_time(df, should_print=False,
                 test_start=pd.to_datetime("30 June 2018 00:00:00")):
    """Adds a group_backwards and days_backwards column to the data
    
    If ```Use the period starting on 30 June 2018 00:00:00, the day after the last date from the transaction files.``` 
    that means the 29th is included, but the 30th not (it's the first day in our test data;

    Also, the first 14 days backwards should be [16-29] June (The 15th should not be included!)

    So we index "group_backwards" which is how many weeks BACKWARDS from test time we have (ie, 0 weeks backwards means we are at TEST TIME). Therefore, 0 doesn't exist for now :)
    """
    df["time"] = pd.to_datetime(df["time"])

    # Make sure we only have data for 2018
    assert (df["time"].dt.year != 2018).sum() == 0
    if should_print:
        print("The first timestamp is", df["time"].min(),
              "and the last is", df["time"].max())

    df["days"] = df["time"].dt.dayofyear

    # Make sure we have data for every single day
    df["days"].unique() == np.arange(1, 181)

    df["days_backwards"] = test_start.dayofyear - df["days"]
    df["group_backwards"] = np.ceil(df["days_backwards"] / 14).astype(int)
    # Make sure we didn't make any mistake - 16th/06 should 1
    assert not (df.set_index("time").loc["16 June 2018 00:00:00":"16 June 2018 23:59:59",
                                             "group_backwards"] != 1).sum()
    # 15th/06 should be 2
    assert not (df.set_index("time").loc["15 June 2018 00:00:00":"15 June 2018 23:59:59",
                                             "group_backwards"] != 2).sum()


def merge_data(orders, items, infos, col="itemID"):
    df = pd.merge(orders, items, on=col, validate="m:1")
    df = pd.merge(df, infos, on=col, validate="m:1")
    return df

def cost_func(target, prediction, simulatedPrice):
    temp = (prediction - np.maximum(prediction - target, 0) * 1.6)
    return np.sum(temp*simulatedPrice)


def promo_detector(orders, aggregation=True, mode=True):
    """
    This function adds a "promotion" column at "orders.csv".
    It verifies if an item of an order is being sold cheaper than it's prices "mode"/"mean". 
    Case affirmative, a '1' will be added in 'promotion' column in the line of the order.
    Parameters: orders -> Orders DataFrame
                aggregation -> Flag that mantains or not the "salesPriceMode" in our returned DataFrame
                True => Return will have the column
                mode -> Decision method flag (Default 'True'). If "True", the function will 
                use the 'mode' of the prices to decide if an item is being sold below it's normal price. 
                If 'False', we'll use the "mean" of the prices.
                
    Returns: our orders Dataframe with 2 new columns ("salesPriceMode" and "promotion")
    """
      
    def agregationMode(x): return x.value_counts().index[0] if mode else 'mean'
    
    # Getting an itemID / salesPriceMode Dataframe
    # salesPriceMode column will store the 
    # 'mean'/'mode' of our items
    pricesAggregated = orders.groupby('itemID').agg(
        salesPriceMode=('salesPrice', agregationMode))

    pricesAggregated['promotion'] = 0
    ordersCopy = orders.copy()
    
    orders_with_promotion = pd.merge(
        ordersCopy, pricesAggregated, how='inner', left_on='itemID', right_on='itemID')
    
    # For every item whose salesPrice is lower than the 'mean'/'mode',
    # we'll attribute 1 to it's position in 'promotion' column
    orders_with_promotion.loc[orders_with_promotion['salesPrice'] <
                                               orders_with_promotion['salesPriceMode'], 'promotion'] = 1
    if (not(aggregation)):
        orders_with_promotion.drop(
            'salesPriceMode', axis=1, inplace=True)
    return orders_with_promotion


def promo_detector_fixed(orders, aggregation=True, mode=True):
    """
    This function adds a "promotion" column at "orders.csv".
    It verifies if an item of an order is being sold cheaper than it's prices "mode"/"mean". 
    Case affirmative, a '1' will be added in 'promotion' column in the line of the order.

    Parameters: orders -> Orders DataFrame
                aggregation -> Flag that mantains or not the "salesPriceMode" in our returned DataFrame
                True => Return will have the column
                mode -> Decision method flag (Default 'True'). If "True", the function will 
                use the 'mode' of the prices to decide if an item is being sold below it's normal price. 
                If 'False', we'll use the "mean" of the prices.
                
    Returns: our orders Dataframe with 2 new columns ("salesPriceMode" and "promotion")
    """
    
    new_df = pd.DataFrame()
      
    def agregationMode(x): return x.value_counts().index[0] if mode else 'mean'
    
    for i in range(1, 14):
        # Getting an itemID / salesPriceMode Dataframe
        # salesPriceMode column will store the 
        # 'mean'/'mode' of our items
        current_agg = orders.loc[orders.group_backwards <= i].groupby(['itemID']).agg(salesPriceMode=('salesPrice', agregationMode))
        
        current_agg['promotion'] = 0
        orders_copy = orders.loc[orders.group_backwards == i].copy()
        
        current_orders_with_promotion = pd.merge(orders_copy, current_agg, how='inner', left_on='itemID', right_on='itemID')
        
        # For every item whose salesPrice is lower than the 'mean'/'mode',
        # we'll attribute 1 to it's position in 'promotion' column
        current_orders_with_promotion.loc[current_orders_with_promotion['salesPrice'] <
                                                       current_orders_with_promotion['salesPriceMode'], 'promotion'] = 1
        
        new_df = pd.concat([new_df, current_orders_with_promotion])
    
    if (not(aggregation)):
        new_df.drop(
            'salesPriceMode', axis=1, inplace=True)
        
    new_df.sort_values(by=['group_backwards', 'itemID'], inplace=True)
    
    return new_df


def promotionAggregation(orders, items, timeConstraint=3, promotionMode='mean', timeScale='group_backwards', salesPriceMode='mean'):
    """The 'promotion' feature is, originally, given by sale. This function aggregates it into the selected
    time scale.
    
    Parameters
    -------------
    orders : A pandas DataFrame with all the sales.
    
    items: A pandas DataFrame with the infos about all items
                
    promotionMode : A pandas aggregation compatible data type; 
                    The aggregation mode of the 'promotion' feature
    timeScale : A String with the name of the column containing the time signature.
                E.g.: 'group_backwards'
    salesPriceMode : A pandas aggregation compatible data type;
                    The aggregation mode of the 'salesPrice' feature
                    
    Return
    -------------
    A new pandas DataFrame grouped by time signature and aggregated by the function passed as argument.     
    """
    

    df = orders.groupby([timeScale, 'itemID'], as_index=False).agg(
        {'order': 'sum', 'promotion': promotionMode, 'salesPrice': salesPriceMode})

    items_copy = items.copy()

    df.rename(columns={'order': 'orderSum', 'promotion': f'promotion_{promotionMode}',
                       'salesPrice': f'salesPrice_{salesPriceMode}'}, inplace=True)
    return pd.merge(df, items_copy, how='left', left_on=['itemID'], right_on=['itemID'])
