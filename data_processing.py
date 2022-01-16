from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
import seaborn as sns

import os

def load_data(data_type='train', without_NAs=True):
    data_path = os.path.join('data', 'data_'+data_type+'.json')
    sequential_data = []
    with open(data_path,'r') as f:
        for line in f:
            sequential_data.append(json.loads(line))
    
    sequential_driver = {} 
    sequential_behavior = {}

    for item in sequential_data:
        # get each user_id for each order
        user_id = item[0]
        # get each order time
        application_time = int(item[1]['order_info']['order_time'])
        # get non-sequential data associated with each order time
        sequential_driver.update({f"{user_id}|{application_time}" : item[1]['order_info']})
        # get sequential data (page view info) associated with each order time
        sub_data = [x for x in item[1]['data']]
        sequential_behavior.update({f"{user_id}|{application_time}":sub_data})
    
    #Load driver into dataframe, seperate user_id and application_time
    driver = pd.DataFrame(sequential_driver).T.reset_index()
    driver['user_id'] = driver['index'].apply(lambda x : x.split('|')[0])
    driver['application_time'] = driver['index'].apply(lambda x : x.split('|')[1])

    #timeseries transformation
    driver['application_date'] = pd.to_datetime(driver['order_time'],unit='ms')
    driver["day_of_week"]=(driver["application_date"].dt.dayofweek+1).astype(str) 
    driver["hour_of_day"]=(driver["application_date"].dt.hour//4+1).astype(str)

    behavior = []
    for user_keys in sequential_behavior:    
        user_id, application_time = user_keys.split("|")
        index=user_keys
        for item in sequential_behavior[user_keys]:    
            subitem = item.copy()
            subitem.update({"user_id":user_id, "application_time":application_time,"index":index})
            behavior.append(subitem)

    behavior = pd.DataFrame(behavior)
    behavior = behavior.sort_values(['user_id', 'application_time', 'pstime'])
    if without_NAs:
        behavior = behavior[~behavior['pid'].isnull()]
    else:
        behavior = behavior[behavior['pid'].isnull()]

    behavior["stay_time"]=(behavior["petime"]-behavior["pstime"])/1000

    behavior = behavior[(behavior["stay_time"]>0) & (behavior['stay_time']<=88)]

    percentile = np.arange(0,1,0.1) + 0.1

    stay_time_cutoff=behavior.stay_time.quantile(percentile).values

    stay_time_map={(value,stay_time_cutoff[key+1]):str(key+2) for key,value in enumerate(stay_time_cutoff) if key<9}

    stay_time_map[(0,1.119)]='1'
    alphabet=['2','3','4','5','6','7','8','9','10','1']
    for key,alpha in zip(stay_time_map.keys(),alphabet):
        stay_time_map[key]=alpha

    def mapping(time,maps):
        for key in maps.keys():
            if key[0]<=time<=key[1]:
                return maps[key]
    
    behavior['stay_time_label']=behavior.stay_time.apply(lambda x:mapping(x,maps=stay_time_map))
    #transform ms to s at the same time, and if this is the first page, assign lag_time of 0
    behavior=behavior.assign(lagg=lambda x:np.where(x.user_id.shift(1)==x.user_id,(x.pstime-x.petime.shift(1))/1000,0))
    #drop where lag_time>99ths quantile value and lag_time<0
    behavior=behavior[(behavior.lagg<=behavior.lagg.quantile(0.94)) & (behavior.lagg>=0)]
    lagg_cutoff=behavior.lagg.quantile(percentile).values
    lagg_map={(value,lagg_cutoff[key+1]):str(key+2) for key,value in enumerate(lagg_cutoff) if key<9}
    lagg_map[(0,0.113)]='1'
    for key,alpha in zip(lagg_map.keys(),alphabet):
        lagg_map[key]=alpha

    behavior['lagg_label']=behavior.lagg.apply(lambda x:mapping(x,maps=lagg_map))

    behavior.drop(columns=['pstime','petime'],inplace=True)

    behavior=behavior.assign(pid_label=lambda x:np.where(x.user_id.shift(1)==x.user_id,x.pid.ne(x.pid.shift(1)).astype(int),2))
    behavior=behavior.assign(sid_label=lambda x:np.where(x.user_id.shift(1)==x.user_id,x.sid.ne(x.sid.shift(1)).astype(int),2))

    features=['pname','stay_time_label','lagg_label','pid_label','sid_label']

    behavior['word']=behavior[features].apply(lambda x:'|'.join(x.values.astype('str')),axis=1)
    behavior=behavior[['word','index','pname']]

    dictionary = pd.DataFrame(behavior.word.unique())
    dictionary.rename(columns={0:'word'},inplace=True)
    dictionary.loc[2736,'word']='[UNK]'
    dictionary.loc[2737,'word']='[PAD]'
    np.savetxt('data/vocab.txt', dictionary['word'], fmt='%s')
    counter=pd.DataFrame(behavior.word.value_counts()).reset_index().rename(columns={'index':'word','word':'counts'})
    low_frequency_words=counter.loc[counter.counts==1].word.values
    behavior.loc[behavior.word.isin(low_frequency_words),'word']='unknown'
    behavior.loc[~behavior.word.isin(dictionary.word.values),'word']='unknown'
    new_df=pd.DataFrame(behavior.groupby('index').apply(lambda x:' '.join(x.word.values)))
    new_df.reset_index(inplace=True)
    new_df.rename(columns={0:'sentence'},inplace=True)
    new_df.reset_index(inplace=True)
    new_df.rename(columns={'level_0':'id'},inplace=True)
    new_df=new_df.merge(driver,how='left',on='index')
    if data_type == 'test':
        new_df['label'] = np.where(new_df['overdue']>5,1,0)
    
    new_df.to_csv('data/'+data_type+'.csv',index=False)

if __name__ == '__main__':
    data_type = 'train'
    data = pd.read_csv('data/'+data_type+'.csv')
    sub_data = data[['sentence','label']]
    max_sentence_len = 0
    sentence_len_list = []
    with open('data/'+data_type+'.txt', 'w', encoding='UTF-8') as f:
        for item in zip(sub_data['sentence'], sub_data['label']):
            sentence_len = len(item[0].split(' '))
            max_sentence_len = max_sentence_len if max_sentence_len > sentence_len else sentence_len
            sentence_len_list.append(sentence_len)
            f.write(item[0] + ',' +str(int(item[1]))+'\n')
    
    print(max_sentence_len)
    print(sum(sentence_len_list)/len(sentence_len_list))
        