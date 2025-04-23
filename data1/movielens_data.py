import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
import pandas as pd
import random
import pickle

class MovielensData(data.Dataset):
    def __init__(self, data_dir=r'data/ref/movielens',
                 stage=None,
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=1682
        self.padding_rating=0
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        sample = {
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': self.cans_num,
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    
    def negative_sampling(self,seq_unpad,next_item):
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

    def check_files(self):
        self.item_id2name=self.get_movie_id2name()
        if self.stage=='train':
            filename="train_data.df"
        elif self.stage=='val':
            filename="val_data.df"
        elif self.stage=='test':
            filename="test_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  
    
    def get_mv_title(self, s):
        sub_list=[", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:]+" "+s.replace(sub_s,"")
        return s

    def get_movie_id2name(self):
        movie_id2name = dict()
        item_path=op.join(self.data_dir, 'u.item')
        with open(item_path, 'r', encoding = "ISO-8859-1") as f:
            for l in f.readlines():
                ll = l.strip('\n').split('|')
                movie_id2name[int(ll[0]) - 1] = self.get_mv_title(ll[1][:-7])
        return movie_id2name
    
    def session_data4frame(self, datapath, movie_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove((self.padding_item_id,self.padding_rating))
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        def seq_to_title(x): 
            return [movie_id2name[x_i[0]] for x_i in x]
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        def next_item_title(x): 
            return movie_id2name[x[0]]
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        def get_id_from_tumple(x):
            return x[0]
        def get_id_from_list(x):
            return [i[0] for i in x]
        train_data['next'] = train_data['next'].apply(get_id_from_tumple)
        train_data['seq'] = train_data['seq'].apply(get_id_from_list)
        train_data['seq_unpad']=train_data['seq_unpad'].apply(get_id_from_list)
        return train_data

if __name__ == "__main__":

    train_data = MovielensData('/data/mcy/LLaRA/hf_data/movielens', 'test')


    with open('/data/mcy/LLaRA/hf_data/movielens/popular_group.pkl', 'rb') as pklfile:
        popular_group_loaded = pickle.load(pklfile)
    with open('/data/mcy/LLaRA/hf_data/movielens/less_popular_group.pkl', 'rb') as pklfile:
        less_popular_group_loaded = pickle.load(pklfile)
        
    pop_keys = [train_data.get_mv_title(key[:-7]) for key in popular_group_loaded.keys()]
    less_pop_keys = [train_data.get_mv_title(key[:-7]) for key in less_popular_group_loaded.keys()]

    print(pop_keys)
    # movie_id2name = train_data.get_movie_id2name()
    # train_data = train_data.session_data4frame('/data/mcy/LLaRA/hf_data/movielens/Test_data.df', movie_id2name)
    # num_rows = train_data.shape[0]
     
    # # Load the item file (u.item)
    # movieList = {}
    # for line in open("/data/mcy/LLaRA/hf_data/movielens/u.item", encoding='ISO-8859-1').readlines():
    #     (movieId, title) = line.split('|')[0:2]
    #     movieList[movieId] = title 
    
    # movie_ratings_count = {}
    # for line in open("/data/mcy/LLaRA/hf_data/movielens/u.data").readlines():
    #     (uid, mid, rating) = line.split('\t')[0:3]
    #     if mid in movieList:
    #         if int(rating) >= 3 and mid not in movie_ratings_count:
    #             movie_ratings_count[mid] = 0
    #         elif int(rating) >= 3:
    #             movie_ratings_count[mid] += 1

    # sorted_movies = sorted(movie_ratings_count.items(), key=lambda x: x[1], reverse=True)
    # midpoint = len(sorted_movies) // 2
    # popular_group = {movieList[movie_id]: movie_id for movie_id, _ in sorted_movies[:midpoint]}
    # less_popular_group = {movieList[movie_id]: movie_id for movie_id, _ in sorted_movies[midpoint:]}
    
    # print('Popular group:', len(popular_group), len(less_popular_group))
    # with open("/data/mcy/LLaRA/hf_data/movielens/popular_group.pkl", 'wb') as pklfile:
    #     pickle.dump(popular_group, pklfile)
    # with open("/data/mcy/LLaRA/hf_data/movielens/less_popular_group.pkl", 'wb') as pklfile:
    #     pickle.dump(less_popular_group, pklfile)

    # with open('/data/mcy/LLaRA/hf_data/movielens/popular_group.pkl', 'rb') as pklfile:
    #     popular_group_loaded = pickle.load(pklfile)

    #     for k in popular_group_loaded.keys():
    #         print(k, popular_group_loaded[k])