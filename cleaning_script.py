'''
Cleaning and processing script for association rules homework

Required files:
https://www.dropbox.com/s/5o0v3uoakkqe8u6/Dillards%20POS.7z?dl=0
    Cleaning and processing uses 'strinfo.csv' and 'trnsact.csv'

Random sample of five stores drawn from the strinfo.csv file
    motivating theory is single stage cluster sampling. Processing power limits
    the number of clusters available for sampling. More details in report.docx

trnsact.csv filtered for observations from randomly sampled stores and 
stype == "P". Do not want to include returns in the analysis. 

Prelim support filtration
    skus with support less than 0.0001 filtered out

skus onehot encoded, group by store, register, saledate, seq, trannum and
summed to generate dataset of unique transactions

If same item was purchased more than once in a transcation sum > 1
Relabeled these occurrences to 1 for compatibility with association rules
algorithm at Prof Klabjan's recommendation

write to data folder as csv.

'''

import pandas as pd
import random
import numpy as np
random.seed(308)

str_cols = ['store', 'city', 'state', 'zip']
strinfo = pd.read_csv('data/strinfo.csv', sep=',',
                      header=None, usecols=range(0, 4), names=str_cols)

trnsact_colnames = ['sku', 'store', 'register', 'trannum', 'seq',
                    'saledate', 'stype']
trnsact_chunk = pd.read_csv('data/trnsact.csv', sep=',', header=None,
                            chunksize=1000000, usecols=range(0, 7),
                            names=trnsact_colnames)

store_samp = random.sample(strinfo.store.tolist(), 5)

chunk_lst = []
for chunk in trnsact_chunk:
    chunk_filt = chunk[(chunk.store.isin(store_samp)) & (chunk.stype == "P")]
    chunk_lst.append(chunk_filt)

trnsact_df = pd.concat(chunk_lst)

sku_counts = trnsact_df.sku.value_counts()
cutoff = 0.0001 * len(trnsact_df.index)
sku_count_filt = sku_counts[sku_counts >= cutoff]

final_skus = sku_count_filt.index.tolist()
trnsact_df_filt = trnsact_df[trnsact_df.sku.isin(final_skus)]

sku_dummy = pd.get_dummies(trnsact_df_filt['sku'], prefix="sku")

trnsact_df_dummy = pd.concat([trnsact_df_filt, sku_dummy], axis=1)

trnsact_df_dummy = trnsact_df_dummy.drop('sku', axis=1)

trnsact_df_grouped = trnsact_df_dummy.groupby(
    ['store', 'register', 'trannum', 'saledate', 'seq']).sum()

for col in trnsact_df_grouped.columns:
    trnsact_df_grouped[col] = np.where(
        trnsact_df_grouped[col] > 1, 1, trnsact_df_grouped[col])

# .nosync because it kept getting saved to cloud and
# bumped out of computer memory
trnsact_df_grouped.to_csv('data/final_df.nosync.csv', index=False)
