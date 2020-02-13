import numpy as np
import pandas as pd
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# full dataframe
sku_df = pd.read_csv('data/final_df.nosync.csv')

# dataframe used for association rules
# only includes transactions that have more than one purchase
# no need to make the computer do more work than it has to
assoc_df = sku_df[sku_df.sum(axis=1) > 1]

sku_cols = ['sku', 'dept', 'classid', 'upc', 'style',
            'color', 'size', 'packsize', 'vendor', 'brand']
skuinfo = pd.read_csv('data/skuinfo.csv', names=sku_cols, usecols=range(10))

# Top 10 Most Sold Products analysis
top10 = sku_df.mean().sort_values(ascending=False)[0:10, ]
top10_df = pd.DataFrame(top10, columns=['pct'])
top10_df['sku'] = [int(i[4:]) for i in top10_df.index]

top10_df = top10_df.set_index('sku').join(skuinfo.set_index('sku'), how='left')

str_cols = ['store', 'city', 'state', 'zip']
strinfo = pd.read_csv('data/strinfo.csv', sep=',',
                      names=str_cols, usecols=range(0, 4))

skst_cols = ['sku', 'store', 'cost', 'retail']
skstinfo = pd.read_csv('data/skstinfo.nosync.csv', names=skst_cols,
                       usecols=range(0, 4))

random.seed(308)
store_samp = random.sample(strinfo.store.tolist(), 5)

skstinfo = skstinfo[skstinfo.store.isin(store_samp)]

sku_pricing = skstinfo.drop('store', axis=1).groupby(['sku']).mean()
sku_pricing['margin'] = sku_pricing.retail - sku_pricing.cost
top10_df = top10_df.join(sku_pricing, how='left')

print("These are the 10 most commonly purchased items")
print(top10_df.filter(['pct', 'brand', 'cost', 'retail', 'margin']))
print("Dillards sells a lot of makeup!")

# association rules begin here

freqItems = apriori(assoc_df, min_support=0.001, use_colnames=True)
assoc_rules = association_rules(freqItems, metric="lift", min_threshold=1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
print('The following ~100 SKUs should be grouped together on the salesfloor')
print(assoc_rules.sort_values(by='lift', ascending=False).iloc[0:100, ])
