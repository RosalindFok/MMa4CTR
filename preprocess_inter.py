r"""
生成微信和抖音数据集的用户-短视频交互二分图,需要与main.go配合使用 (见README.md)
"""

import os
import numpy as np
import json
import rich.progress
from tqdm import tqdm
import time
import csv
from load_path import graph_path, wechat_dataset_path, tiktok_dataset_path, temp_path, tiktok, wechat, finish, like
from collections import Counter

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

if __name__ == '__main__':
    """抖音数据预处理 track1"""
    if os.path.exists(tiktok_dataset_path):
        print('=======Now is processing Tiktok======')
        
        ########## 处理用户-短视频交互 ##########
        """ reserve_samples.txt保存全样本的一个较小的子集  该文件由main.go生成 """
        if os.path.exists(os.path.join(temp_path, 'reserve_samples.txt')):
            uid_arr,item_id_arr,finish_arr,like_arr,device_arr,time_arr=[],[],[],[],[],[]
            uid_col,item_id_col,finish_col,like_col,device_col,time_col=[i for i in range(6)]
            with rich.progress.open(os.path.join(temp_path, 'reserve_samples.txt'), 'r') as f:
                for line in f:
                    line = line.split(' ')
                    uid_arr.append(int(line[uid_col]))
                    item_id_arr.append(int(line[item_id_col]))
                    finish_arr.append(int(line[finish_col]))
                    like_arr.append(int(line[like_col]))
                    device_arr.append(int(line[device_col]))
                    time_arr.append(int(line[time_col]))
            assert len(uid_arr)==len(item_id_arr)==len(finish_arr)==len(like_arr)==len(device_arr)==len(time_arr)
            print(f'There are {len(uid_arr)} interactions now.') # 49,550,347
            
            # 生成交互字典
            inter_dict = {} # {uid : key, [item_id, finish, like, time] : value}
            for uid, _ in zip(uid_arr, tqdm(range(len(uid_arr)))):
                inter_dict[uid] = []

            for (index, value), _ in zip(enumerate(uid_arr), tqdm(range(len(uid_arr)))):
                inter_dict[value] += [[item_id_arr[index],finish_arr[index],like_arr[index],device_arr[index],time_arr[index]]]
            
            # 写tiktok.inter文件
            print(f'Now is writing tiktok.inter...')
            start = time.time()
            inter_tiktok_path = os.path.join('.','baseline','dataset', tiktok)
            if not os.path.exists(inter_tiktok_path):
                os.mkdir(inter_tiktok_path)
            with open(os.path.join('.','baseline','dataset', tiktok, 'tiktok.inter'), 'w') as f:
                f.write('user_id:token\titem_id:token\ttimestamp:float\n')
                for (uid, value), _ in zip(inter_dict.items(), tqdm(range(len(inter_dict)))):
                    for group in value: # group = [item_id, finish, like, device, time]
                        f.write(str(uid)+'\t'+str(group[0])+'\t'+str(group[-1])+'\n')
            end = time.time()
            print(f'It took {round((end-start)/60, 4)} minutes to write tiktok.inter' )

            # 保存二分图
            if not os.path.exists(graph_path):
                os.mkdir(graph_path)
            finish_content, like_content = [],[] # [[uid, item_id, click], ...]
            device_content = [] # [[uid, device], ...]
            for (uid, value), _ in zip(inter_dict.items(), tqdm(range(len(inter_dict)))):
                for group in value: # group = [item_id, finish, like, device, time]
                    finish_content.append([uid, group[0], group[1]])
                    like_content.append([uid, group[0], group[2]])
                    device_content.append([uid, group[3]])
            assert len(finish_content) == len(like_content) == len(device_content)
            # 嵌套列表去重
            finish_content = [list(i) for i in set(tuple(j) for j in finish_content)]
            like_content = [list(i) for i in set(tuple(j) for j in like_content)]
            device_content = [list(i) for i in set(tuple(j) for j in device_content)]
            save_bipartite_graph(os.path.join(graph_path, tiktok+'_'+finish), finish_content)            
            save_bipartite_graph(os.path.join(graph_path, tiktok+'_'+like), like_content)            
            save_bipartite_graph(os.path.join(graph_path, tiktok+'_device'), device_content)
            
        # 不存在已经筛选过的数据(生成将交由main.go处理的数据)
        else:
            """ 用户交互 final_track1_train.txt """
            (uid_arr, user_city_arr, item_id_arr, author_id_arr, # 用户id 用户城市id 作品id 作者id
            item_city_arr, channel_arr, finish_arr, like_arr,    # 作品城市id 观看渠道 是否浏览完 是否点赞
            music_id_arr, device_arr, time_arr, duration_time_arr# 音乐id 设备id 观看时间 作品时长
                                                ) = [],[],[],[],[],[],[],[],[],[],[],[]
            (uid_col, user_city_col, item_id_col, author_id_col,
            item_city_col, channel_col, finish_col, like_col, 
            music_id_col, device_col, time_col, duration_time_col) = [i for i in range(12)]

            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

            """ temp.txt仅保留uid item_id author finish like device time 这7个字段的信息 """
            # 不存在temp.txt则从头读取
            if not os.path.exists(os.path.join(temp_path, 'temp.txt')): 
                with rich.progress.open(os.path.join(tiktok_dataset_path,'final_track1_train.txt'), 'r') as f:
                    for line in f:
                        line = line.split('\t')
                        uid_arr.append(int(line[uid_col]))
                        item_id_arr.append(int(line[item_id_col]))
                        author_id_arr.append(int(line[author_id_col]))
                        finish_arr.append(int(line[finish_col])) # 1和0
                        like_arr.append(int(line[like_col]))     # 1和0
                        device_arr.append(int(line[device_col]))
                        time_arr.append(int(line[time_col]))
                assert len(uid_arr) == len(item_id_arr) == len(author_id_arr) == len(finish_arr) == len(like_arr) == len(device_arr) == len(time_arr) 
                # temp.txt不存在则写入
                with open(os.path.join(temp_path, 'temp.txt'), 'w') as f:
                    for index, _ in zip(range(len(uid_arr)), tqdm(range(len(uid_arr)))):
                        # uid item_id author_id finish like device time
                        write_str = str(uid_arr[index])+' '+str(item_id_arr[index])+' '+str(author_id_arr[index])+' '+str(finish_arr[index])+' '+str(like_arr[index])+' '+str(device_arr[index])+' '+str(time_arr[index])+'\n'
                        f.write(write_str)
                print(f'temp.txt has been written. preprocess.py will Run again to generate reserve_uid.txt')

            # 直接从temp.txt中读入 加快速度 需要保存留下的uid  
            else: 
                uid_arr,item_id_arr,author_id_arr,finish_arr,like_arr,device_arr,time_arr=[],[],[],[],[],[],[]
                uid_col,item_id_col,author_id_col,finish_col,like_col,device_col,time_col=[i for i in range(7)]
                with rich.progress.open(os.path.join(temp_path, 'temp.txt'), 'r') as f:
                    for line in f:
                        line = line.split(' ')
                        uid_arr.append(int(line[uid_col]))
                        item_id_arr.append(int(line[item_id_col]))
                        author_id_arr.append(int(line[author_id_col]))
                        finish_arr.append(int(line[finish_col]))
                        like_arr.append(int(line[like_col]))
                        device_arr.append(int(line[device_col]))
                        time_arr.append(int(line[time_col]))
                assert (len(uid_arr) == len(item_id_arr) ==  len(author_id_arr) == len(finish_arr) == 
                    len(like_arr) == len(device_arr) == len(time_arr))

            #  """对该数据集进行全局统计"""
            # 用户交互长度
            uid_inter_num = Counter(uid_arr) # dict类型 {uid : key, uid在uid_arr中出现的次数 : value}
            uid_inter_cnt = Counter(list(uid_inter_num.values())) # {用户的交互条数 : key, 具有该交互条数的用户数目 : value}
            inter_num, inter_num_cnt = list(uid_inter_cnt.keys()), list(uid_inter_cnt.values()) 
            assert len(inter_num) == len(inter_num_cnt)
            # 最长交互 21302; 最短交互 1; 平均交互 53846
            print(f'The max interation = {max(inter_num)}; the min = {min(inter_num)}; the avg = {(np.sum(np.array(inter_num)*np.array(inter_num_cnt))) / len(inter_num)}')
            if not os.path.exists(os.path.join('..', 'temp')):
                os.mkdir(os.path.join('..', 'temp'))
            with open(os.path.join('..', 'temp','user_inter_num.json'), 'w') as f:
                f.write(json.dumps(uid_inter_cnt))
            # 物品流行度
            item_inter_num = Counter(item_id_arr)
            item_inter_cnt = Counter(list(item_inter_num.values()))
            inter_num, inter_num_cnt = list(item_inter_cnt.keys()), list(item_inter_cnt.values()) 
            assert len(inter_num) == len(inter_num_cnt)
            # 最长被交互 149500; 最短被交互 1; 平均被交互 25899 
            print(f'The max interation = {max(inter_num)}; the min = {min(inter_num)}; the avg = {(np.sum(np.array(inter_num)*np.array(inter_num_cnt))) / len(inter_num)}')
            if not os.path.exists(os.path.join('..', 'temp')):
                os.mkdir(os.path.join('..', 'temp'))
            with open(os.path.join('..', 'temp','item_popul_num.json'), 'w') as f:
                f.write(json.dumps(item_inter_cnt))
            
            # 全部数据 -- 用户 : 636281; 短视频 : 27342248; 交互 : 275,855,531
            print(f'There are {len(uid_arr)} interactions in total. {len(set(uid_arr))} users and {len(set(item_id_arr))} micro-videos.')
            # 2个点击指标 finish和like
            pos = sum(finish_arr)
            print(f'In finish, Neg / Pos = {round((len(finish_arr)-pos) / pos, 4)}') # 全 : 2.5322
            pos = sum(like_arr)
            print(f'In like, Neg / Pos = {round((len(like_arr)-pos) / pos, 4)}') # 全 : 61.2809

            # 保留交互长度>=N的用户 : 剩下的用户数目M. 1500 : 35645; 2000 : 17450; 3000 : 4997; 4000 : 1678
            min_length = 4000
            reserve_uid, sum = [], 0
            for (key, value), _ in zip(uid_inter_num.items(), tqdm(range(len(uid_inter_num)))):
                if value >= min_length:
                    reserve_uid.append(key)
                    sum+=value
            print(f'There are {len(reserve_uid)} uid are left now.') # 17450
            reserve_uid.sort()
            print(f'There are {sum} interaction left.') # 2000 : 49,550,347; 3000 : 19,909,447; 4000 : 8,622,897

            # reserve_uid.txt 需要保留的uid 交由main.go处理
            with open(os.path.join(temp_path,'reserve_uid.txt'), 'w') as f:
                for uid,_ in zip(reserve_uid, tqdm(range(len(reserve_uid)))):
                    f.write(str(uid) + '\n')

            print(f'reserve_uid.txt has been done. Plead go to main.go.')
            exit(0)
        # item_id.txt 保存item_id_arr 用户TikTok多模态的处理
        with open(os.path.join(temp_path, 'item_id.txt'), 'w') as f:
            for item_id,_ in zip(item_id_arr, tqdm(range(len(item_id_arr)))):
                f.write(str(item_id) + '\n')
    else:
        print("TikTok2019 dataset does not exist!")

########################################------------------------------########################################
########################################------------------------------########################################
########################################------------------------------########################################

    """微信数据预处理"""
    if os.path.exists(wechat_dataset_path):
        print('=======Now is processing Wechat Channel======')
        # 7项评价指标的权重
        """user_action.csv"""
        # 字段: userid feedid date_ device read_comment comment like play stay click_avatar forward follow favorite 
        # device为设备类型id, play为视频播放时长
        with open(os.path.join(wechat_dataset_path,"user_action.csv"),'r') as file:
            lines = list(csv.reader(file))
            head = lines[0]
            (userid_col,feedid_col,read_comment_col,comment_col,like_col,click_avatar_col,forward_col,follow_col,
                favorite_col,device_col,play_col) = (head.index('userid'), head.index('feedid'), head.index('read_comment'),
                head.index('comment'), head.index('like'), head.index('click_avatar'), head.index('forward'), 
                head.index('follow'), head.index('favorite'), head.index('device'), head.index('play'))
            # 7个指标 每个指标构建一个二分图: 二分图每行为一个用户 每列为一个短视频 每个元素代表该用户和该短视频在这个指标下是否有交互
            # 由于是稀疏图, 因此只保存元素为1的节点 (userid, feeid)
            # 对于对短视频没有任何点击的用户 是否需要保存, 在二分图中就是一个孤立的节点
            if not (os.path.exists(graph_path)):
                os.makedirs(graph_path)
            (read_comment_content,no_read_comment_content,comment_content,no_comment_content,like_content, 
                no_like_content, click_avatar_content, no_click_avatar_content, forward_content, no_forward_content, 
                follow_content, no_follow_content, favorite_content, no_favorite_content
                , device_content, userid_arr, feedid_arr, new_lines) = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
            click = []
            lines.pop(0)
            # 7个点击率指标各自的权重 read_comment:4 like:3 click_avtar:2 forward:1 favorite:1 comment:1 follow:1
            total_weight = 4+3+2+1+1+1+1
            (read_comment_weight, like_weight, click_avatar_weight, forward_weight, favorite_weight, 
            comment_weight, follow_weight) = [i/total_weight for i in [4,3,2,1,1,1,1]]

            print(f'Reading user_action.csv: ')
            for line, i in zip(lines, tqdm(range(len(lines)+1))):
                assert len(line) == len(head)
                # play的值非0 才说明该用户与短视频的交互是有效的
                # if line[play_col] != '0': # 效果不如人意
                if True:
                    # device作为用户的side info
                    device_content.append([int(line[userid_col]), int(line[device_col])]) # [[userid, device], ...]
                    # 将7个指标按照权重进行线性组合得到一个指标
                    node = (int(line[userid_col]), int(line[feedid_col]))
                    userid_arr.append(int(line[userid_col]))
                    feedid_arr.append(int(line[feedid_col]))
                    click.append([node[0], node[1], read_comment_weight*int(line[read_comment_col])+
                                like_weight*int(line[like_col])+click_avatar_weight*int(line[click_avatar_col])+
                                forward_weight*int(line[forward_col])+favorite_weight*int(line[favorite_col])+
                                comment_weight*int(line[comment_col])+follow_weight*int(line[follow_col])]) # [[userid, feedid, click], ...]
                    # 7个权重独自统计
                    read_comment_content.append(node) if(line[read_comment_col] == '1') else no_read_comment_content.append(node)
                    comment_content.append(node) if(line[comment_col] == '1') else no_comment_content.append(node)
                    like_content.append(node) if(line[like_col] == '1') else no_like_content.append(node)
                    click_avatar_content.append(node) if(line[click_avatar_col] == '1') else no_click_avatar_content.append(node)
                    forward_content.append(node) if(line[forward_col] == "1") else no_forward_content.append(node)
                    follow_content.append(node) if(line[follow_col] == '1') else no_follow_content.append(node)
                    favorite_content.append(node) if(line[favorite_col] == '1') else no_favorite_content.append(node)
                    new_lines.append(node)

            # 打印正负样本比例 统计人数和短视频数
            print(f'There are {len(set(userid_arr))} users and {len(set(feedid_arr))} micro-videos')

            print(f'There are {len(new_lines)} interaction.')

            read_comment_ratio = round(len(no_read_comment_content)/len(read_comment_content),3)
            comment_ratio = round(len(no_comment_content)/len(comment_content),3)
            like_ratio = round(len(no_like_content)/len(like_content),3)
            click_avatar_ratio = round(len(no_click_avatar_content)/len(click_avatar_content),3)
            forward_ratio = round(len(no_forward_content)/len(forward_content),3)
            follow_ratio = round(len(no_follow_content)/len(follow_content),3)
            favorite_ratio = round(len(no_favorite_content)/len(favorite_content),3)
            print(f'In read_comment Raw datas, Neg / Pos = {read_comment_ratio}') # 25.87
            print(f'In comment Raw datas, Neg / Pos = {comment_ratio}')           # 2406.245
            print(f'In like Raw datas, Neg / Pos = {like_ratio}')                 # 36.491
            print(f'In click_avatar Raw datas, Neg / Pos = {click_avatar_ratio}') # 124.931
            print(f'In forward Raw datas, Neg / Pos = {forward_ratio}')           # 236.349
            print(f'In follow Raw datas, Neg / Pos = {follow_ratio}')             # 1355.267
            print(f'In favorite Raw datas, Neg / Pos = {favorite_ratio}')         # 715.507
            # 398.167623
            print(f'The weighted Neg / Pos = {read_comment_weight*read_comment_ratio + comment_weight*comment_ratio+ like_weight*like_ratio + click_avatar_weight*click_avatar_ratio + forward_weight*forward_ratio + follow_weight*follow_ratio + favorite_weight*favorite_ratio}')
            # 阈值 threshold 大于等于该threshold设为1 小于则设为0
            threshold = 0.5
            pos_len = len([i for i in click if i[2] >= threshold])
            print(f'Threshold set as {threshold}, the New Neg / Pos = {round((len(click)-pos_len)/pos_len,3)}') # 389.228
            click_length = len(click)
            for i, _ in zip(range(click_length), tqdm(range(click_length))):
                click[i][2] = 1 if click[i][2] >= threshold else 0

            # 进行正确性断言判定
            assert len(click) == len(device_content) == len(new_lines)
            assert pos_len == len([i for i in click if i[2] == 1])
            assert ((len(read_comment_content)+len(no_read_comment_content))==(len(comment_content)+len(no_comment_content))==
                    (len(like_content)+len(no_like_content))==(len(click_avatar_content)+len(no_click_avatar_content))==
                    (len(forward_content)+len(no_forward_content))==(len(follow_content)+len(no_follow_content))==
                    (len(favorite_content)+len(no_favorite_content))==len(device_content))

            # 嵌套列表去重
            click = [list(i) for i in set(tuple(j) for j in click)]
            device_content = [list(i) for i in set(tuple(j) for j in device_content)]
            assert len(click) == len(set([str(i) for i in click]))
            assert len(device_content) == len(set([str(i) for i in device_content]))
            print(f'There are {len(click)} interactions in CLICK and {len(device_content)} records in DEVICE.')
            
            # 将userid-device和userid-feedid-click保存为两个图
            save_bipartite_graph(os.path.join(graph_path, wechat+'_device'), device_content)
            save_bipartite_graph(os.path.join(graph_path, wechat+'_click'), click)  
    else:
        print("Error: Wechat dataset does not exist!")

exit(0)