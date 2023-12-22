r"""
    划分训练集/验证集/测试集 比例为8:1:1
    生成单一模态矩阵
"""
import os,time
import orjson as json 
import numpy as np
from tqdm import tqdm
import rich.progress
from  utils import save_npy, get_userid_itemid, get_multimodal_dict, pca_process
from load_path import graph_path, matrix_path, user_embedding_path, each_modal_arr, tiktok, wechat, finish,like

# 微信数据集 -- 读取多模态内容并保存到矩阵
def gen_matrix(user_feed : dict, tag : str, click_tag = None)->None:
    tag = tag.lower()
    click_tag = click_tag.lower() if not click_tag is None else None
    # 分别处理3种模态
    for modal_tag in each_modal_arr:
        # 获取用户embedding
        if tag == wechat:
            embedding_file_path=tag+'_'+modal_tag+'_user_embeddings.json'
        elif tag == tiktok and click_tag == finish or click_tag == like:
            embedding_file_path=tag+'_'+click_tag+'_'+modal_tag+'_user_embeddings.json'
        else:
            print(f'Error {tag} is wrong. ')

        with rich.progress.open(os.path.join(user_embedding_path, embedding_file_path), 'r') as f:
            userid_embedding = json.loads(f.read())

        # 获取模态信息
        modal_dict = get_multimodal_dict(tag+'_'+modal_tag)

        if len(list(modal_dict.values())[0]) > 128: # 太长的模态进行降维
            # wechat: visual 192; acoustic 242; textual 73
            # tiktok: visual 71; acoustic 10; textual 22
            modal_value = list(modal_dict.values())
            modal_value = pca_process(np.array(modal_value), 128)
            assert len(modal_dict.keys()) == len(modal_value)
            for key, value, _ in zip(modal_dict.keys(), modal_value, range(len(modal_value))):
                modal_dict[key] = value
        
        # Attention: 统计绘图的时候不要删除任何用户
        interaction_num = []
        for userid in user_feed:
            interaction_num.append(len(user_feed[userid]))
        print(f'The max interaction number is {max(interaction_num)}; the min is {min(interaction_num)}; the avg is {sum(interaction_num) / len(interaction_num)}')
        
        # 生成矩阵
        matrix = []
        cnt, user_length = 0, len(user_feed)
        start = time.time()
        for userid in user_feed:
            for feedid_click, _ in zip(user_feed[userid], tqdm(range(len(user_feed[userid])))):
                # 每个feedid_click = [feedid, click]
                # 第一项为userid 用于一些简单的统计
                if feedid_click[0] in modal_dict: # 抖音数据集中 部分交互的item没有模态记录 不纳入训练矩阵
                    arr = [userid] + userid_embedding[str(userid)] + modal_dict[feedid_click[0]] + [feedid_click[1]] 
                    matrix.append(arr)
            cnt += 1
            print(f'{cnt} / {user_length} user has Processed.')
        end = time.time()
        print(f'{tag} : {modal_tag} matrix took {round((end-start)/60,3)} minutes.')
        # 保存矩阵
        if tag == wechat:
            save_npy(os.path.join(matrix_path, tag+'_'+modal_tag+'_matrix'), np.array(matrix))
        elif tag == tiktok:
            save_npy(os.path.join(matrix_path, tag+'_'+click_tag+'_'+modal_tag+'_matrix'), np.array(matrix))


if __name__ == '__main__':
    # 存放用于训练的矩阵
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)

    # 获取用户-短视频和用户-设备二分图 形式为dict 以userid为key进行索引
    if os.path.exists(graph_path):
        user_feed, _ = get_userid_itemid(graph_path, tag = wechat)
        user_finish, user_like, user_device = get_userid_itemid(graph_path, tag = tiktok)
    else:
        print(graph_path,'does not exits!')

    # 获取模态信息
    gen_matrix(user_feed, tag=wechat)
    gen_matrix(user_finish, tag=tiktok, click_tag=finish)
    gen_matrix(user_like, tag=tiktok, click_tag=like)
    exit(0)