# -*- coding: utf-8 -*-
"""

Define multistep predictability function, which aligns the predictability metrics
to the words they correspond to. 

"""

import pandas as pd

def multistep_predictability(multistep_predictions, n_predict):
    

    # Melt the DataFrame from wide to long
    df_long = pd.wide_to_long(
        multistep_predictions,
        stubnames=['pred_token_t+', 'pred_word_t+', 'match_t+', 'conf_t+', 'rank_t+'],
        i=['word_idx', 'word', 'hftoken', 'token_id'],
        j='timestep'
    ).reset_index()

    # Rename columns for clarity
    df_long.rename(columns={
        'pred_token_t+': 'pred_token',
        'pred_word_t+': 'pred_word',
        'match_t+': 'match',
        'conf_t+': 'conf',
        'rank_t+': 'rank'
    }, inplace=True)
    
    # Create new columns, pred_rank and pred_conf, that align the rank and
    # conf columns so that they refer to how well a word was predicted by
    # its preceding word(s)
    df_long['pred_rank'] = None
    df_long['pred_conf'] = None
    for i in range(len(df_long)):    
        index_offset = df_long.loc[i, 'timestep'] * n_predict
        if i - index_offset >= 0:
            pred_rank = df_long.loc[i  -  index_offset, 'rank'] 
            pred_conf = df_long.loc[i  -  index_offset, 'conf'] 
        else:
            pred_rank = pd.NA
            pred_conf = pd.NA
        df_long.at[i, 'pred_rank'] = pred_rank
        df_long.at[i, 'pred_conf'] = pred_conf
    
    return df_long