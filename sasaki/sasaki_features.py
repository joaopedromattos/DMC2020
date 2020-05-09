import os
import pandas as pd
import numpy as np


orders= pd.read_csv('../main/datasets/1.0v/orders.csv', sep = '|')
items= pd.read_csv('../main/datasets/1.0v/items.csv', sep = '|')

import sys
sys.path.append("../main")

from utils import read_data, process_time, merge_data


def remove_itemID_not_used(items, orders):
    """removes from item.csv the rows with item without sales"""
    id_used = orders.itemID.unique()

    boolean = [np.isin(idd,id_used) for idd in items.itemID]
    items= items[boolean] 
    
    return items




def add_feature_promotion(items, orders):
    """ add to orders.csv and items.csv:
        modeSalesPrice: the sales price mode of rows with same itemID
        
        add to orders.csv:
        difModa: error between modeSalesPrice and salesPrice
        """
    
    def _get_mode_sales_price(id):
        sales_id_certo = orders[orders['itemID'] == id]
        mode = sales_id_certo['salesPrice'].mode()

        #a moda pode ter mais de um valor, 
        #para torna-la um unico foi escolhido arbitrariamente a mediana
        return mode.median()
    
    
    items['modeSalesPrice'] =items['itemID'].map(_get_mode_sales_price )
    
    modeSalesPrice = items[['itemID','modeSalesPrice']]
    orders = pd.merge(orders, modeSalesPrice, on='itemID')
    
    
    orders['difModa'] = orders['salesPrice'] - orders['modeSalesPrice']

    return items, orders
    

def agregating_by_week(items, orders,add_zero_salues=False, time_processed=True, promotion_added=True):
    """ agregate orders.csv by week , itemID and salesPrice
        if add_zero_salues is true, add rows of itemID in every week, even if the total sales is zero
        """
        
    
    if ( time_processed == False):
        process_time(orders)
    
    if ( promotion_added == False):
        items, orders = add_feature_promotion(items, orders)
        
    
    orders["week_backward"] = np.ceil(orders["days_backwards"] / 7).astype(int)
    
    #removing the first week, it has 5 days only
    orders = orders[orders["week_backward"]!=26]
    

    #agregating by week
    orders_w = orders.groupby(['itemID','salesPrice','week_backward']).agg({'order' : ['sum'], 
                                                               'group_backwards' : ['mean'],
                                                               'modeSalesPrice' : ['mean'], 
                                                               'difModa' : ['mean']})
    orders_w = orders_w.reset_index()
    orders_w.columns = ['itemID','salesPrice','week_backward','order','group_backwards','modeSalesPrice','difModa']
    
    
    if(add_zero_salues):
        weeks_database = orders['week_backward'].unique()
        new_rows = []
        
        for idd in items['itemID'].unique():
            orders_id = orders_w[orders_w.itemID == idd]
            example = orders_id.iloc[0]
    
            #finding weeks without itemID sales
            weeks_id = orders_id['week_backward'].unique()
            weeks_without_id = np.setdiff1d(weeks_database , weeks_id)
    
            #creating new row
            for w in weeks_without_id:
                new_rows.append({'itemID':idd, 
                                 'salesPrice':example['modeSalesPrice'], 
                                 'week_backward':w, 
                                 'order':0,
                                 'group_backwards':example['group_backwards'], 
                                 'modeSalesPrice':example['modeSalesPrice'], 
                                 'difModa':example['difModa']
                                })

        orders_w = orders_w.append(new_rows) 

        
    return orders_w


def add_feature_position_month(orders, time_processed=True, add_feature_week=False):
    """ add to orders.csv:
        posM_f_group: position in the month of the first day of the group
        posM_m_group: position of the midle day of the group
        posM_l_group: position of the last day
        
        if add_feature_week:
        also add the 3 features above but for the week
        """
    
    if time_processed == False:
        process_time(orders)
            
    #getting first and last date of every group
    groups = orders.groupby(['group_backwards'], as_index=False).agg({'time':['min', 'max']})
    groups.columns = ['group_backwards','first_date_g','last_date_g']

    #ok, a pretty ugly way to get the midle day of the group
    groups['median'] = groups['first_date_g'] + pd.DateOffset(days=7)

    
    # position in the month of the first day of the group
    groups['posM_f_group'] = groups['first_date_g'].map( lambda x : x.day)

    groups['posM_m_group'] = groups['median'].map( lambda x : x.day)
    
    groups['posM_l_group'] = groups['last_date_g'].map( lambda x : x.day)

    #left join with orders
    groups = groups[['group_backwards','posM_f_group','posM_m_group','posM_l_group']]
    orders = pd.merge(orders, groups, on='group_backwards', how='left')
    
    
    #do the same thing but for the week
    if add_feature_week:
        #getting first and last date of every week
        weeks = orders.groupby(['week_backwards'], as_index=False).agg({'time':['min', 'max']})
        weeks.columns = ['week_backwards','first_date_w','last_date_w']
        weeks['median'] = weeks['first_date_w'] + pd.DateOffset(days=3)

        #position in the month of the first day of the week
        weeks['posM_f_week'] = weeks['first_date_w'].map( lambda x : x.day)

        weeks['posM_m_week'] = weeks['median'].map( lambda x : x.day)

        weeks['posM_l_week'] = weeks['last_date_w'].map( lambda x : x.day)


        #left join with orders
        weeks = weeks[['week_backwards','posM_f_week','posM_m_week','posM_l_week']]
        orders = pd.merge(orders, weeks,on='week_backwards',how='left')
        
    return orders