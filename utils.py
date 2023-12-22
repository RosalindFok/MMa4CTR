r"""
工具类
"""

import time, torch, os, rich.progress
import numpy as np
import orjson as json
from tqdm import tqdm
from typing import Tuple
from sklearn.decomposition import PCA
from load_path import graph_path, modal_path, wechat, tiktok

# Embedding 每个单词扩充到3维向量; 
def embedding_process(arr : np.array)->np.array:
    print(f'Embedding Start!')
    start = time.time()
    # 二维转一维
    if arr.ndim == 2:
        word_arr = set([i for j in arr for i in j])
    elif arr.ndim == 1:
        word_arr = set(arr)
    # 词的个数 (最大值要加1啊!!)
    num_embeddings = max(word_arr)+1
    # embedding_dim = int(len(word_arr)**0.25) # 每个词语/即原来的一个正整数 转为一个长度固定的向量 向量中的每个浮点数的小数点后保留了4位
    embedding_dim = int(len(word_arr)**0.25) if int(len(word_arr)**0.25) <= 64 else 64 
    embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    arr_embedding = embedding(torch.LongTensor(arr))
    if arr.ndim == 2:
        result = np.reshape(arr_embedding.detach().numpy(), (len(arr), len(arr[0])*embedding_dim))
    elif arr.ndim == 1:
        result = np.reshape(arr_embedding.detach().numpy(), (len(arr), embedding_dim))
    end = time.time()
    print(f'Embedding took {end-start} seconds. The dim = {len(result[0])}')
    assert len(result) == len(arr)
    return result

# PCA降维 默认保留 90%的主要成分
def pca_process(arr : np.array, dim = 0.9)->list[float]:
    print(f'PCA Start!')
    start = time.time()
    pca_dim = PCA(n_components=dim)
    arr = pca_dim.fit_transform(arr).tolist()
    result = []
    # 只保留4位小数
    for x, _ in zip(arr, tqdm(range(len(arr)))):
        result.append([float('{:.4f}'.format(i)) for i in x])
    end = time.time()
    print(f'PCA took {round((end-start)/60, 3)} minutes. The dim = {len(result[0])}')
    assert len(result) == len(arr)
    return result

# 删除指定路径的文件
def delete_file(file : str):
    if os.path.exists(file):
        os.remove(file)

# 获取用户-短视频点击率和用户-设备二分图  返回字典方便查找
def get_userid_itemid(dir : str,  tag : str, delete : bool = True):
    # 微信 click.npy和device.npy
    if tag.lower() == wechat:
        file = os.path.join(dir, tag+'_click.npy')
        # 正样本(有点击行为的样本) 负样本(无点击行为的样本)
        content = (np.load(os.path.join(graph_path, file), allow_pickle=True)).tolist()
        user_feed = {}
        for x, _ in zip(content, tqdm(range(len(content)))):
            user_feed[x[0]] = []
        for x, _ in zip(content, tqdm(range(len(content)))):
            user_feed[x[0]].append([x[1], x[2]])
        delete_keys = []
        # 如果一个用户刷不到100条短视频 就把TA删除了 保留16107个用户
        for x, _ in zip(user_feed, tqdm(range(len(user_feed)))):
            if len(user_feed[x]) < 100: # 暂定100
                delete_keys.append(x)
        # 删除 
        if delete:
            for x in delete_keys:
                del user_feed[x]

        file = os.path.join(dir, tag+'_device.npy')
        content = np.load(os.path.join(graph_path, file), allow_pickle=True)
        user_device = {}
        for x, _ in zip(content, tqdm(range(len(content)))):
            user_device[x[0]] = [x[1]] # Attention!!: 单个int转tensor必须将int转为list 否则值会改变
        print(len(user_device),len(user_feed))
        assert len(user_device) >= len(user_feed)
        # user_feed: {userid : [[feedid, clickOrNot], [feedid, clickOrNot]], userid : [...], ...}
        # user_device: {userid : device, ...}
        return user_feed, user_device
    # tiktok finish.npy和like.npy和device.npy
    elif tag.lower() == tiktok:
        # finish
        finsh_file = os.path.join(dir, tag+'_finish.npy')
        content = (np.load(os.path.join(graph_path, finsh_file), allow_pickle=True)).tolist()
        user_finish = {}
        for group in content:
            user_finish[group[0]] = []
        for group in content:
            user_finish[group[0]].append([group[1], group[-1]])
        
        # like
        like_file = os.path.join(dir, tag+'_like.npy')
        content = (np.load(os.path.join(graph_path, like_file), allow_pickle=True)).tolist()
        user_like = {}
        for group in content:
            user_like[group[0]] = []
        for group in content:
            user_like[group[0]].append([group[1], group[-1]])

        #device
        device_file = os.path.join(dir, tag+'_device.npy')
        content = (np.load(os.path.join(graph_path, device_file), allow_pickle=True)).tolist()
        user_device = {}
        for group in content:
            user_device[group[0]] = [group[-1]]
        
        assert len(user_finish) == len(user_like) == len(user_device)
        # user_finish: {uid : [[item_id, click], [item_id, click], ...]}
        # user_like  : {uid : [[item_id, click], [item_id, click], ...]}
        # user_device: {uid : device}
        return user_finish, user_like, user_device

    else:
        print(f'Error: {tag} is wrong tag. ')
        exit(1)

# 获取各个模态的向量 返回字典方便查找
def get_multimodal_dict(modal_tag : str) -> dict:
    print(f'Start to get {modal_tag} information...')
    start = time.time()
    with rich.progress.open(os.path.join(modal_path, modal_tag+'.json'), 'rb') as f:
        data = json.loads(f.read())
    end = time.time()
    print(f'{modal_tag} json loaded! It took {round((end - start)/60, 3)} minutes.')
    # 建立以feedid为key 到模态embedding的索引
    modal_dict = {}
    start = time.time()
    ##### 微信 #####
    if modal_tag == 'wechat_visual':
        for x, _ in zip(data, tqdm(range(len(data)))):
            # ocr : 192 ; ocr_char : 208
            modal_dict[x['feedid']] = x['ocr'] # 192
    elif modal_tag == 'wechat_acoustic':
        for x, _ in zip(data, tqdm(range(len(data)))):
            # asr : 170 ; asr_char : 219 ; bgm_singer_id : 11 ; bgm_song_id : 12
            modal_dict[x['feedid']] = x['asr_char'] + x['bgm_singer_id'] + x['bgm_song_id'] # 242
    elif modal_tag == 'wechat_textual':    
        for x, _ in zip(data, tqdm(range(len(data)))):
            # description : 43 ; description_char : 52 ;
            # manual_keyword_list :  18 ; machine_keyword_list : 16 ; manual_tag_list : 14 ; machine_tag_list : 14
            modal_dict[x['feedid']] = x['description'] + x['machine_keyword_list'] + x['machine_tag_list'] # 73
    ##### 抖音 #####
    elif modal_tag.startswith('tiktok_'):
        for x, _ in zip(data, tqdm(range(len(data)))):
            # visual_dim : 71; acoustic_dim : 10; textual_dim : 22
            modal_dict[int(list(x.keys())[0])] = list(x.values())[0] 
    else:
        print(f'Error: Check modal_tag = {modal_tag}!')
        exit(1)
    
    return modal_dict

# 保存到npy格式
def save_npy(path : str, content : np.array)->None:
    print(f'Start to save npy...')
    start = time.time()
    np.save(path+'.npy', content, allow_pickle=True)
    end = time.time()
    print('--------------------------------')
    print(f'{path}, " Saved!" It took {end - start} seconds to complete.')
    print('--------------------------------')
    return

# 对二维矩阵 按列归一化
def norm_as_column(matrix : list[list[float]])->list[list[float]]:
    result = []
    # 矩阵转置
    matrix = list(map(list, zip(*matrix)))
    # 每行归一化
    for row in matrix:
        tmp = []
        max_value, min_value = max(row), min(row)
        divisor = max_value - min_value
        for value,_ in zip(row, tqdm(range(len(row)))):
            tmp.append((value - min_value)/divisor if not divisor == 0 else 0)
        result.append(tmp)
    # 矩阵转置
    result = list(map(list, zip(*result)))
    return result