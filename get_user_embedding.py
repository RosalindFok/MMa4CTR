r"""
根据用户的交互和多模态内容 生成用户的embedding
"""
import os, time
import numpy as np
import json
from tqdm import tqdm
from utils import get_userid_itemid, get_multimodal_dict, norm_as_column
from load_path import graph_path, user_embedding_path, each_modal_arr, tiktok, wechat, finish, like
from sklearn.preprocessing import MinMaxScaler

if not os.path.exists(user_embedding_path):
    os.mkdir(user_embedding_path)

""" adjust the length of user's multi modal embedding """
# 用户embedding中模态部分的长度 不包含与辅助信息的拼接
user_embedding_modal_part_length = 21

# 平均池化 将M×1的list 映射到 N×1的list
def average_pooling(arr : list, dst_length : int = 64)->list:
    result = []
    if len(arr) < dst_length:
        arr += [.0] * (dst_length-len(arr))
    pool_length = len(arr) - dst_length + 1
    for index in range(dst_length):
        tmp = arr[index : index + pool_length]
        result.append(sum(tmp)/len(tmp))
    assert len(result) == dst_length
    return result

if __name__ == '__main__':
    """ 微信 """
    # 获取三个模态的字典
    visual_modal_dict = get_multimodal_dict(wechat+'_visual')
    acoustic_modal_dict = get_multimodal_dict(wechat+'_acoustic')
    textual_modal_dict = get_multimodal_dict(wechat+'_textual')
    assert visual_modal_dict.keys() == acoustic_modal_dict.keys() == textual_modal_dict.keys()

    # 将模态全部归一化, 以免交叉时有影响. 归一化前转为numpy数组可以提升速度. 每个值都是[0,1]之间的浮点数
    visual_modal_value = list(visual_modal_dict.values())
    acoustic_modal_value = list(acoustic_modal_dict.values())
    textual_modal_value = list(textual_modal_dict.values())

    visual_modal_value = MinMaxScaler().fit_transform(np.array(visual_modal_value)) # 192
    acoustic_modal_value = MinMaxScaler().fit_transform(np.array(acoustic_modal_value)) # 242 
    textual_modal_value = MinMaxScaler().fit_transform(np.array(textual_modal_value)) # 73

    assert len(visual_modal_value) == len(acoustic_modal_value) == len(textual_modal_value) == len(visual_modal_dict)

    # 重新制作字典 其值已经归一化
    for modal_dict, value_list in zip([visual_modal_dict, acoustic_modal_dict, textual_modal_dict], 
                                    [visual_modal_value, acoustic_modal_value, textual_modal_value]):
        assert len(modal_dict) == len(value_list)
        for key, value, _ in zip(list(modal_dict.keys()), value_list, tqdm(range(len(value_list)))):
            modal_dict[key] = value

    # 每个模态下有一个用户embedding 最后保存为json格式文件
    for modal_dict, modal_tag in zip([visual_modal_dict,acoustic_modal_dict,textual_modal_dict],each_modal_arr):
        # 获取用户-短视频和用户-设备二分图 形式为dict 以userid为key进行索引
        if os.path.exists(graph_path):
            user_all_embeddings = {}
            user_feed, user_device = get_userid_itemid(graph_path, tag=wechat)
            for userid, _ in zip(list(user_feed.keys()), tqdm(range(len(user_feed)))):
                pos_click_list, neg_click_list, user_embedding = [], [], np.array([.0]*len(list(modal_dict.values())[0]))

                for feedid_click in user_feed[userid]:
                    if feedid_click[1] == 1: # 正交互样本
                        pos_click_list.append(feedid_click[0]) # 添加feedid
                    elif feedid_click[1] == 0: # 负交互样本
                        neg_click_list.append(feedid_click[0]) # 添加feedid
                    else:
                        print(f'Error: Wrong with click_graph')
                        exit(1)
                assert len(pos_click_list)+len(neg_click_list) == len(user_feed[userid])

                # 对于用户user pos_click_list为其点击的短视频集合, neg_click_list为其没有点击的短视频集合 (二分图的边分为正边和负边)
                # 某个用户, 其embedding由其正负邻居的向量来表示. pos+1, neg-1
                # 这个±1的加权 是否类似于某种卷积
                if len(pos_click_list) > 0:
                    for feedid in pos_click_list:
                        user_embedding += np.array(modal_dict[feedid])
                if len(neg_click_list) > 0:
                    for feedid in neg_click_list:
                        user_embedding -= np.array(modal_dict[feedid])
                # 淡化交互数目的影响 平均一次
                divisor = np.array([len(pos_click_list)+len(neg_click_list)+.0] * len(list(modal_dict.values())[0]))
                user_embedding /= divisor
                
                # 对每个N×1的embedding进行平均池化, 缩短其长度
                user_embedding = user_embedding.tolist()
                user_embedding = average_pooling(user_embedding, user_embedding_modal_part_length)
                # user的embedding由其邻居节点(交互过的短视频)的模态向量和辅助信息向量拼接而成
                user_embedding += user_device[userid]
                user_all_embeddings[str(userid)] = user_embedding
            
            # 对所有的embedding按列归一化: 好处是保留同维度上的数量大小关系
            embedding_values = list(user_all_embeddings.values())
            embedding_values = norm_as_column(embedding_values)
            assert len(embedding_values) == len(user_all_embeddings)
            for key, user_embedding, _ in zip(list(user_all_embeddings.keys()), embedding_values, tqdm(range(len(embedding_values)))):
                user_all_embeddings[key] = user_embedding

            # 保存embedding信息
            start = time.time()
            file_path = os.path.join(user_embedding_path, wechat+'_'+modal_tag+'_user_embeddings.json')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(user_all_embeddings))
            end = time.time()
            print(f'It took {round((end-start),4)} seconds to save {file_path}')
        else:
            print(graph_path,'does not exits!')

    """ 抖音 """
    # 获取三个模态的字典
    visual_modal_dict = get_multimodal_dict(tiktok+'_visual') 
    acoustic_modal_dict = get_multimodal_dict(tiktok+'_acoustic')
    textual_modal_dict = get_multimodal_dict(tiktok+'_textual') 

    # 将模态全部归一化, 以免交叉时有影响. 归一化前转为numpy数组可以提升速度. 每个值都是[0,1]之间的浮点数
    visual_modal_value = list(visual_modal_dict.values())
    acoustic_modal_value = list(acoustic_modal_dict.values())
    textual_modal_value = list(textual_modal_dict.values())

    visual_modal_value = MinMaxScaler().fit_transform(np.array(visual_modal_value)) # 71
    acoustic_modal_value = MinMaxScaler().fit_transform(np.array(acoustic_modal_value)) # 10 
    textual_modal_value = MinMaxScaler().fit_transform(np.array(textual_modal_value)) # 22

    # 重新制作字典 其值已经归一化
    for modal_dict, value_list in zip([visual_modal_dict, acoustic_modal_dict, textual_modal_dict], 
                                    [visual_modal_value, acoustic_modal_value, textual_modal_value]):
        assert len(modal_dict) == len(value_list)
        for key, value, _ in zip(list(modal_dict.keys()), value_list, tqdm(range(len(value_list)))):
            modal_dict[key] = value
    
    # 每个模态下有一个用户embedding 最后保存为json格式文件
    for modal_dict, modal_tag in zip([visual_modal_dict,acoustic_modal_dict,textual_modal_dict],each_modal_arr):
        # 获取用户-短视频和用户-设备二分图 形式为dict 以uid为key进行索引
        if os.path.exists(graph_path):
            user_all_embeddings = {}
            user_finish, user_like, user_device = get_userid_itemid(graph_path, tag=tiktok)

            # 有些item_id不在modal_dict中需要删除之
            for user_inter in [user_finish, user_like]:
                for uid, _ in zip(user_inter, tqdm(range(len(user_inter)))):
                    inter_arr = user_inter[uid]
                    delete_itemid = []
                    for item_click in inter_arr:
                        item_id = item_click[0]
                        if not item_id in modal_dict: # 该交互的短视频没有多模态信息
                            delete_itemid.append(item_click)
                    if not len(delete_itemid) == 0:
                        for item_click in delete_itemid:
                            inter_arr.remove(item_click)
                    if not len(inter_arr) == 0:
                        user_inter[uid] = inter_arr
                    else:
                        print(uid)
                        exit(1)

            # finish and like
            for user_item_clike, click_tag in zip([user_finish, user_like],[finish,like]):
                for uid, _ in zip(list(user_item_clike.keys()), tqdm(range(len(user_item_clike)))):
                    pos_click_list, neg_click_list, user_embedding = [], [], np.array([.0]*len(list(modal_dict.values())[0]))
                    inter_arr = user_item_clike[uid]
                    for itemid_click in inter_arr:
                        if itemid_click[-1] == 1: # 正交互样本
                            pos_click_list.append(itemid_click[0]) # 添加itemid
                        elif itemid_click[-1] == 0: # 负交互样本
                            neg_click_list.append(itemid_click[0]) # 添加itemid
                        else:
                            print(f'Error: Wrong with click_graph')
                            exit(1)
                    assert len(pos_click_list)+len(neg_click_list) == len(inter_arr)

                    # 对于用户uid pos_click_list为其点击的短视频集合, neg_click_list为其没有点击的短视频集合 (二分图的边分为正边和负边)
                    # 某个用户, 其embedding由其正负邻居的向量来表示. pos+1, neg-1
                    # 这个±1的加权 是否类似于某种卷积
                    if len(pos_click_list) > 0:
                        for item_id in pos_click_list:
                            user_embedding += np.array(modal_dict[item_id])
                    if len(neg_click_list) > 0:
                        for item_id in neg_click_list:
                            user_embedding -= np.array(modal_dict[item_id])
                    
                    # 淡化交互数目的影响 平均一次
                    divisor = np.array([len(pos_click_list)+len(neg_click_list)+.0] * len(list(modal_dict.values())[0]))
                    user_embedding /= divisor
                    # 对每个N×1的embedding进行平均池化, 缩短其长度
                    user_embedding = user_embedding.tolist()
                    user_embedding = average_pooling(user_embedding, user_embedding_modal_part_length)
                    # user的embedding由其邻居节点(交互过的短视频)的模态向量和辅助信息向量拼接而成
                    user_embedding += user_device[uid]
                    user_all_embeddings[str(uid)] = user_embedding

                # 对所有的embedding按列归一化: 好处是保留同维度上的数量大小关系
                embedding_values = list(user_all_embeddings.values())
                embedding_values = norm_as_column(embedding_values)
                assert len(embedding_values) == len(user_all_embeddings)
                for key, user_embedding, _ in zip(list(user_all_embeddings.keys()), embedding_values, tqdm(range(len(embedding_values)))):
                    user_all_embeddings[key] = user_embedding

                # 保存embedding信息
                start = time.time()
                file_path = os.path.join(user_embedding_path, tiktok+'_'+click_tag+'_'+modal_tag+'_user_embeddings.json')
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(json.dumps(user_all_embeddings))
                end = time.time()
                print(f'It took {round((end-start),4)} seconds to save {file_path}')
        else:
            print(graph_path,'does not exits!')
    
    exit(0)