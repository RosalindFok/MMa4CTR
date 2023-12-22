r"""
处理微信和抖音数据集的短视频多模态信息
"""

import os
import csv
import numpy as np
import orjson as json
import rich.progress
from tqdm import tqdm
import time
from keras_preprocessing import sequence
from load_path import modal_path, wechat_dataset_path, tiktok_dataset_path, temp_path, tiktok, wechat
from utils import pca_process, embedding_process

# 保存用户-短视频二分图\用户-设备二分图
def save_bipartite_graph(path : str, content : list)->None: 
    start = time.time()
    np.save(path+'.npy', content)
    end = time.time()
    print('--------------------------------')
    print('Length of the graph = ', len(content))
    print(f'{path}, " Saved!" It took {end - start} seconds to complete.')
    print('--------------------------------')
    return

# 保存每个短视频的多模态信息
def save_modal_json(path : str, content : list[dict])->None:
    start = time.time()
    with open(path, 'wb') as file:
        file.write(json.dumps(content))
    end = time.time()
    print('--------------------------------')
    print('Length of the modal = ', len(content))
    print(f'{path}, " Saved!" It took {end - start} seconds to complete.')
    print('--------------------------------')
    return 

# 将使用空格分隔的字符串数组的每一个元素化为list str转为int
def str2list(arr : list[str])->list[list]:
    slice = []
    print(f'Spliting...')
    for x, i in zip(arr, tqdm(range(0,len(arr)+1))):
        slice.append(x.split(' '))
    result = []
    for x in slice:
        tmp = []
        if len(x) == 1 and x[0] == '':
            result.append(tmp)
        else:
            for y in x:
                tmp.append(int(y))
            result.append(tmp)
    assert len(result) == len(arr)
    return result

# 将英文分号分割的多个元素转换为list str转为int
def arr2list(arr : list[str])->list[list]:
    slice = []
    print(f'Spliting...')
    for x, i in zip(arr, tqdm(range(0,len(arr)+1))):
        tmp = x.split(';')
        for i in range(len(tmp)):
            tmp[i] = tmp[i].split(" ")[0]
        slice.append(tmp)
    result = []
    for x in slice:
        tmp = []
        if len(x) == 1 and x[0] == '':
            result.append(tmp)
        else:
            for y in x:
                tmp.append(int(y))
            result.append(tmp)
    assert len(result) == len(arr)
    return result

# 将list[str]转为list[int] 缺失值先补0
def strList2intList(arr : list[str])->list[int]:
    result = []
    for x in arr:
        if x == '':
            result.append(0)
        else:
            result.append(int(x))
    assert len(result) == len(arr)
    return result

# 二分查找
def binary_search(list,item)->int:
    # 列表的头和尾，代表着数组范围的最小和最大
    low = 0
    high = len(list) - 1

    # 当找到item的时候，low是小于high，也有可能相等
    while low <= high:
        mid = (low + high)//2
        # 取数组的中间值
        guess = list[mid]
        # 如果中间值等于索引值，那么就返回中间值的下标
        if guess == item:
            return mid
        # 如果中间值>索引值，因为不包含中间值，所以最大范围high=中间值的下标往左移1位
        if guess > item:
            high = mid - 1
        # 如果中间值<索引值，因为不包含中间值，所以最小范围low=中间值的下标往右移1位
        else:
            low = mid + 1
    return -1

if __name__ == '__main__':
    """微信数据预处理"""
    if os.path.exists(wechat_dataset_path):
        print('=======Now is processing Wechat Channel======')
        """feed_info.csv"""
        # 字段: feedid authorid videoplayseconds       
        # 视觉模态字段: ocr ocr_char
        # 声音模态字段: asr asr_char bgm_song_id bgm_singer_id
        # 文本模态字段: description description_char manual_keyword_list machine_keyword_list manual_tag_list machine_tag_list
        with open(os.path.join(wechat_dataset_path,"feed_info.csv"),'r') as file:
            lines = list(csv.reader(file))
            head = lines[0]
            feedid_col = head.index('feedid')
            ocr_col = head.index('ocr')
            ocr_char_col = head.index('ocr_char')
            asr_col = head.index('asr')
            asr_char_col = head.index('asr_char')
            bgm_song_id_col = head.index('bgm_song_id')
            bgm_singer_id_col = head.index('bgm_singer_id')
            description_col = head.index('description')
            description_char_col = head.index('description_char')
            manual_keyword_list_col = head.index('manual_keyword_list')
            machine_keyword_list_col = head.index('machine_keyword_list')
            manual_tag_list_col = head.index('manual_tag_list')
            machine_tag_list_col = head.index('machine_tag_list')
            # 3个模态 每个模态保存一份数据 对于列表型数据需要提前转换好
            if not os.path.exists(modal_path):
                os.makedirs(modal_path)
            (feedid_arr, ocr_arr, ocr_char_arr, asr_arr, asr_char_arr, bgm_song_id_arr, bgm_singer_id_arr, description_arr, 
                description_char_arr, manual_keyword_list_arr, machine_keyword_list_arr, manual_tag_list_arr, 
                machine_tag_list_arr) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
            lines.pop(0)
            print(f'Reading feed_info.csv: ')
            for line, i in zip(lines, tqdm(range(0,len(lines)+1))):
                feedid_arr.append(int(line[feedid_col]))
                ocr_arr.append(line[ocr_col])
                ocr_char_arr.append(line[ocr_char_col])
                asr_arr.append(line[asr_col])
                asr_char_arr.append(line[asr_char_col])
                bgm_song_id_arr.append(line[bgm_song_id_col])
                bgm_singer_id_arr.append(line[bgm_singer_id_col])
                description_arr.append(line[description_col])
                description_char_arr.append(line[description_char_col])
                manual_keyword_list_arr.append(line[manual_keyword_list_col])
                machine_keyword_list_arr.append(line[machine_keyword_list_col])
                manual_tag_list_arr.append(line[manual_tag_list_col])
                machine_tag_list_arr.append(line[machine_tag_list_col])
            assert len(feedid_arr)==len(ocr_arr)==len(ocr_char_arr)==len(asr_arr)==len(asr_char_arr)\
                ==len(bgm_song_id_arr)==len(bgm_singer_id_arr)==len(description_arr)==len(description_char_arr)\
                ==len(manual_keyword_list_arr)==len(machine_keyword_list_arr)==len(manual_tag_list_arr)==len(machine_tag_list_arr)
            # 保存各个模态的信息: feedid-field1-field2... 不定长的字段需要处理.
            # 保存模态信息为json格式,每一行代表一个短视频,格式为:{feedid : id, field1 : value1, field2 : value2,...}
            # 视觉模态
            print(f'Processing Visual modal...')
            ocr_arr = str2list(ocr_arr)
            ocr_char_arr = str2list(ocr_char_arr)
            # 使用ocr和ocr_char互相填充空白值
            ocr_length = len(ocr_arr)
            for index, _ in zip(range(ocr_length), tqdm(range(ocr_length))):
                if len(ocr_arr[index]) == 0 and len(ocr_char_arr[index]) != 0:
                    ocr_arr[index] = ocr_char_arr[index]
                if len(ocr_arr[index]) != 0 and len(ocr_char_arr[index]) == 0:
                    ocr_char_arr[index] = ocr_arr[index]
            # 末尾补零 向量对齐
            ocr_arr = sequence.pad_sequences(ocr_arr, padding = 'post') # 每个5198维
            ocr_char_arr = sequence.pad_sequences(ocr_char_arr, padding = 'post') # 每个7796维  

            # 使用PCA进行降维 保留90%的信息
            ocr_arr = pca_process(ocr_arr) # 192
            ocr_char_arr = pca_process(ocr_char_arr) # 208  

            modal_content = []
            for feedid, ocr, ocr_char, _ in zip(feedid_arr, ocr_arr, ocr_char_arr, tqdm(range(len(feedid_arr)))):
                modal_content.append({'feedid' : feedid, 'ocr' : ocr, 'ocr_char' : ocr_char})
            save_modal_json(os.path.join(modal_path, wechat+'_visual.json'),modal_content)

            # 声音模态
            print(f'Processing Acoustic modal...')
            asr_arr = str2list(asr_arr)
            asr_char_arr = str2list(asr_char_arr)
            # 使用asr和asr_char相互补充缺失值
            asr_length = len(asr_arr)
            for index, _ in zip(range(asr_length), tqdm(range(asr_length))):
                if len(asr_arr[index]) == 0 and len(asr_char_arr[index]) != 0:
                    asr_arr[index] = asr_char_arr[index]
                if len(asr_arr[index]) != 0 and len(asr_char_arr[index]) == 0:
                    asr_char_arr[index] = asr_arr[index]
            # 末尾补零 向量对齐
            asr_arr = sequence.pad_sequences(asr_arr, padding = 'post') # 每个都是1003维
            asr_char_arr = sequence.pad_sequences(asr_char_arr, padding = 'post') # 每个都是1401维

            # 使用PCA进行降维 保留90%的信息
            asr_arr = pca_process(asr_arr) # 170
            asr_char_arr = pca_process(asr_char_arr) # 219  

            # 没有背景音乐也算作一个类型 其值为0
            bgm_singer_id_arr = strList2intList(bgm_singer_id_arr) # 1维
            bgm_song_id_arr = strList2intList(bgm_song_id_arr) # 1维

            bgm_singer_id_arr = embedding_process(np.array(bgm_singer_id_arr)) # 11
            bgm_singer_id_arr = bgm_singer_id_arr.tolist()
            bgm_song_id_arr = embedding_process(np.array(bgm_song_id_arr)) # 12
            bgm_song_id_arr = bgm_song_id_arr.tolist()

            modal_content = []
            for feedid, asr, asr_char, bgm_singer_id, bgm_song_id, _ in zip(feedid_arr, asr_arr, asr_char_arr, 
                                                                bgm_singer_id_arr, bgm_song_id_arr, tqdm(range(len(feedid_arr)))):
                modal_content.append({'feedid' : feedid, 'asr' : asr, 'asr_char' : asr_char, 
                    'bgm_singer_id' : bgm_singer_id, 'bgm_song_id' : bgm_song_id})
            save_modal_json(os.path.join(modal_path, wechat+'_acoustic.json'),modal_content)    

            # 文本模态
            print(f'Processing Textual modal...')
            description_arr = str2list(description_arr)
            description_char_arr = str2list(description_char_arr)
            # 使用description和description_char相互补充缺失值
            description_length = len(description_arr)
            for index, _ in zip(range(description_length), tqdm(range(description_length))):
                if len(description_arr[index]) == 0 and len(description_char_arr[index]) != 0:
                    description_arr[index] = description_char_arr[index]
                if len(description_arr[index]) != 0 and len(description_char_arr[index]) == 0:
                    description_char_arr[index] = description_arr[index]
            # 末尾补零 向量对齐
            description_arr = sequence.pad_sequences(description_arr, padding = 'post') # 每个622维
            description_char_arr = sequence.pad_sequences(description_char_arr, padding = 'post') # 每个899维   

            # 使用PCA进行降维 保留90%的信息
            description_arr = pca_process(description_arr) # 43
            description_char_arr = pca_process(description_char_arr) # 52

            manual_keyword_list_arr = arr2list(manual_keyword_list_arr)
            machine_keyword_list_arr = arr2list(machine_keyword_list_arr)
            # 使用manual_keyword_list_arr和machine_keyword_list_arr互相填充缺失值
            keyword_length = len(manual_keyword_list_arr)
            for index, _ in zip(range(keyword_length), tqdm(range(keyword_length))):
                if manual_keyword_list_arr[index] == 0 and machine_keyword_list_arr[index] != 0:
                    manual_keyword_list_arr[index] = machine_keyword_list_arr[index]
                if manual_keyword_list_arr[index] != 0 and machine_keyword_list_arr[index] == 0:
                    machine_keyword_list_arr[index] = manual_keyword_list_arr[index]
            # 末尾补零 向量对齐
            manual_keyword_list_arr = sequence.pad_sequences(manual_keyword_list_arr, padding = 'post') #每个18维
            machine_keyword_list_arr = sequence.pad_sequences(machine_keyword_list_arr, padding = 'post') # 每个16维    

            # 使用PCA进行降维 保留90%的信息
            manual_keyword_list_arr = manual_keyword_list_arr.tolist()
            machine_keyword_list_arr = machine_keyword_list_arr.tolist()

            manual_tag_list_arr = arr2list(manual_tag_list_arr)
            machine_tag_list_arr = arr2list(machine_tag_list_arr)
            # 使用manual_tag_list_和machine_tag_list相互补充缺失值   
            tag_length = len(manual_tag_list_arr)
            for index, _ in zip(range(tag_length), tqdm(range(tag_length))):
                if len(manual_tag_list_arr[index]) == 0 and len(machine_tag_list_arr[index]) != 0:
                    manual_tag_list_arr[index] = machine_tag_list_arr[index]
                if len(manual_tag_list_arr[index]) != 0 and len(machine_tag_list_arr[index]) == 0:
                    machine_tag_list_arr[index] = manual_tag_list_arr[index]
            manual_tag_list_arr = sequence.pad_sequences(manual_tag_list_arr, padding = 'post') # 每个14维
            machine_tag_list_arr = sequence.pad_sequences(machine_tag_list_arr, padding = 'post') # 每个14维

            # 使用PCA进行降维 保留90%的信息
            manual_tag_list_arr = manual_tag_list_arr.tolist()
            machine_tag_list_arr = machine_tag_list_arr.tolist()

            modal_content = []
            for feedid, description, description_char, manual_keyword_list, machine_keyword_list, manual_tag_list, machine_tag_list, _ in zip(
            feedid_arr, description_arr, description_char_arr, manual_keyword_list_arr, machine_keyword_list_arr, manual_tag_list_arr, machine_tag_list_arr, tqdm(range(len(feedid_arr)))
            ):
                modal_content.append({'feedid':feedid, 'description':description,'description_char':description_char,
                'manual_keyword_list':manual_keyword_list,'machine_keyword_list':machine_keyword_list,
                'manual_tag_list':manual_tag_list,'machine_tag_list':machine_tag_list})
            save_modal_json(os.path.join(modal_path, wechat+'_textual.json'),modal_content)
    else:
        print("Error: Wechat dataset does not exist!")  
        exit(1) 
    
########################################------------------------------########################################
########################################------------------------------########################################
########################################------------------------------########################################

    """抖音数据预处理 track1"""
    if os.path.exists(tiktok_dataset_path):
        print('=======Now is processing Tiktok======')
        ########## 处理TikTok多模态信息 ##########
        item_id_arr = []
        with rich.progress.open(os.path.join(temp_path, 'item_id.txt'), 'r') as f:
            for line in f:
                item_id_arr.append(int(line))
        item_id_arr = set(item_id_arr)
        # 获取多模态文件
        files = os.listdir(tiktok_dataset_path)
        visual_files = [x for x in files if 'video' in x]
        acoustic_files = [x for x in files if 'audio' in x]
        textual_files = [x for x in files if 'title' in x]
        if not os.path.exists(modal_path):
            os.makedirs(modal_path)

        # 视觉特征 
        visual_content = [] # [{item_id : key, video_feature_dim_128 : value}, {...}, ...]
        for file in visual_files:
            file = os.path.join(tiktok_dataset_path, file)
            ids, values = [], []
            with rich.progress.open(file, 'r') as f:
                print(f'Process {file} ...')
                for line in f:
                    line = json.loads(line) 
                    item_id = int(line['item_id'])
                    feature = line['video_feature_dim_128']
                    # 只保留出现在item_id_arr中的短视频的信息
                    if item_id in item_id_arr:
                        if len(feature) == 128:
                            ids.append(item_id)
                            values.append(feature)
                        elif len(feature) < 128:
                            feature += [.0]*(128-len(feature))
                            ids.append(item_id)
                            values.append(feature)
                        else:
                            print(f'Error: visual feature dim > 128.')
                            exit(1)
            values = pca_process(values) # 均是71维
            assert len(values[0]) == 71 and len(values) == len(ids)
            for (index, item_id), _ in zip(enumerate(ids), tqdm(range(len(ids)))):
                visual_content.append({str(item_id) : values[index]})
        save_modal_json(os.path.join(modal_path, tiktok+'_visual.json'), visual_content)   

        # 声音特征 
        acoustic_content = [] # [{item_id : key, audio_feature_128_dim : value}, {...}, ...]
        for file in acoustic_files:
            file = os.path.join(tiktok_dataset_path, file)
            ids, values = [],[]
            with rich.progress.open(file, 'r') as f:
                print(f'Process {file} ...')
                for line in f:
                    line = json.loads(line)
                    item_id = int(line['item_id'])
                    feature = line['audio_feature_128_dim']

                    # 只保留出现在item_id_arr中的短视频的信息
                    if item_id in item_id_arr:
                        if len(feature) == 128:
                            ids.append(item_id)
                            values.append(feature)
                        elif len(feature) < 128:
                            feature += [.0]*(128-len(feature))
                            ids.append(item_id)
                            values.append(feature)
                        else:
                            print(f'Error: acoustic feature dim > 128.')
                            exit(1)
            values = pca_process(values, 10) # 均是10维, 如果保留90%会仅有2维
            assert len(values[0]) == 10 and len(values) == len(ids)
            for (index, item_id), _ in zip(enumerate(ids), tqdm(range(len(ids)))):
                acoustic_content.append({str(item_id) : values[index]})
        save_modal_json(os.path.join(modal_path, tiktok+'_acoustic.json'), acoustic_content)   
        
        # 文本特征 
        textual_content = [] # [{item_id : key, title_features : value}, {...}, ...]
        for file in textual_files:
            file = os.path.join(tiktok_dataset_path, file)
            ids, values = [],[]
            with rich.progress.open(file, 'r') as f:
                print(f'Process {file} ...')
                for line in f:
                    line = json.loads(line)
                    item_id = int(line['item_id'])
                    feature_dict = line['title_features']
                    feature = []
                    for char, times in feature_dict.items():
                        feature += [int(char)] * times
                    # 只保留出现在item_id_arr中的短视频的信息
                    if item_id in item_id_arr:
                        ids.append(item_id)
                        values.append(feature if len(feature) > 0 else [0])
            values = sequence.pad_sequences(values, padding='post')
            values = pca_process(values) # 均是22维
            assert len(values[0]) == 22 and len(values) == len(ids)
            for (index, item_id), _ in zip(enumerate(ids), tqdm(range(len(ids)))):
                textual_content.append({str(item_id) : values[index]})
        save_modal_json(os.path.join(modal_path, tiktok+'_textual.json'), textual_content)
    else:
        print("TikTok2019 dataset does not exist!")
    
    exit(0)