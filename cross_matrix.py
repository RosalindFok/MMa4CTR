r"""
    划分训练集/验证集/测试集 比例为8:1:1
    将userid和3种模态各自交叉
    将3种模态之间两两交叉
"""
import os,time
import json 
from tqdm import tqdm
import numpy as np
import rich.progress
from  utils import pca_process
from utils import save_npy, get_userid_itemid, get_multimodal_dict
from sklearn.preprocessing import MinMaxScaler
from load_path import graph_path, matrix_path, user_embedding_path, double_modal_cross_arr, triple_modal_cross_arr, double_modal_concat_arr, triple_modal_concat_arr, tiktok, wechat, finish, like

""" adjust how much matrix will be made """
# only_vat_concat = True  # 仅生成三种模态拼接的矩阵 
only_vat_concat = False # 生成所有的拼接、交叉矩阵

""" add user_embedding or not"""
user_embedding_exit = True # 进行了user embedding
# user_embedding_exit = False # 完全消融掉了usesr embedding   

# 特征交叉 2
def cross_array(arr_1 : np.array, arr_2 : np.array) :
    result_hadamard = []
    len_1, len_2 = len(arr_1[0]), len(arr_2[0])
    # 将元素更长的那个用PCA进行降维 使其内部元素和较短的相同
    if len_1 > len_2:
        arr_1 = np.array(pca_process(arr_1, len_2)) 
    else:
        arr_2 = np.array(pca_process(arr_2, len_1))
    assert len(arr_1[0]) == len(arr_2[0])
    
    for x, y, _ in zip(arr_1, arr_2, tqdm(range(len(arr_1)))):
        result_hadamard.append((x*y).tolist())
    return result_hadamard

# 特征交叉 3
def cross_array_tri(arr_1 : np.array, arr_2 : np.array, arr_3 : np.array):
    result_hadamard = []
    len_1, len_2, len_3 = len(arr_1[0]), len(arr_2[0]), len(arr_3[0])
    if min(len_1, len_2, len_3) == len_1:
        arr_2 = np.array(pca_process(arr_2, len_1))
        arr_3 = np.array(pca_process(arr_3, len_1))
    elif min(len_1, len_2, len_3) == len_2:
        arr_1 = np.array(pca_process(arr_1, len_2))
        arr_3 = np.array(pca_process(arr_3, len_2))
    elif min(len_1, len_2, len_3) == len_3:
        arr_1 = np.array(pca_process(arr_1, len_3))
        arr_2 = np.array(pca_process(arr_2, len_3))
    assert len(arr_1[0]) == len(arr_2[0]) == len(arr_3[0])

    for x,y,z,_ in zip(arr_1, arr_2, arr_3, tqdm(range(len(arr_1)))):
        result_hadamard.append((x*y*z).tolist())
    return result_hadamard

# 特征拼接 2
def concat_array(arr_1 : np.array, arr_2 : np.array):
    arr = np.concatenate((arr_1, arr_2),axis = 1)
    return arr.tolist()

# 特征拼接 3
def concat_array_tri(arr_1 : np.array, arr_2 : np.array , arr_3 : np.array):
    arr = np.concatenate((arr_1, arr_2), axis=1)
    arr = np.concatenate((arr, arr_3), axis=1)
    return arr.tolist()

if __name__ == '__main__':
    # 存放用于训练的矩阵
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)

    for dataset_tag in [wechat, tiktok]:  
        
        # 获取用户-短视频和用户-设备二分图 形式为dict 以userid为key进行索引
        if not os.path.exists(graph_path):
            print(f'Error: {graph_path} does not exits!')
            exit(1)

        # ########## wechat ##########
        if dataset_tag == wechat:
            user_feed, user_device = get_userid_itemid(graph_path, tag=dataset_tag)
            # 获取用户的embedding 
            visual_embedding_file = dataset_tag+'_visual_user_embeddings.json'
            acoustic_embedding_file = dataset_tag+'_acoustic_user_embeddings.json'
            textual_embedding_file = dataset_tag+'_textual_user_embeddings.json'
        
            with rich.progress.open(os.path.join(user_embedding_path, visual_embedding_file), 'r') as f:
                visual_userid_embedding = json.loads(f.read())
            with rich.progress.open(os.path.join(user_embedding_path, acoustic_embedding_file), 'r') as f:
                acoustic_userid_embedding = json.loads(f.read())
            with rich.progress.open(os.path.join(user_embedding_path, textual_embedding_file), 'r') as f:
                textual_userid_embedding = json.loads(f.read())

            # 获取模态信息
            visual_modal_dict = get_multimodal_dict(dataset_tag+'_visual')
            acoustic_modal_dict = get_multimodal_dict(dataset_tag+'_acoustic')
            textual_modal_dict = get_multimodal_dict(dataset_tag+'_textual')
            assert visual_modal_dict.keys() == acoustic_modal_dict.keys() == textual_modal_dict.keys()      
            # 将模态全部归一化, 以免交叉时有影响
            visual_modal_value = list(visual_modal_dict.values())
            acoustic_modal_value = list(acoustic_modal_dict.values())
            textual_modal_value = list(textual_modal_dict.values())     
            visual_modal_value = MinMaxScaler().fit_transform(visual_modal_value) # 192
            acoustic_modal_value = MinMaxScaler().fit_transform(acoustic_modal_value) # 242 
            textual_modal_value = MinMaxScaler().fit_transform(textual_modal_value) # 73        
            # PCA降维
            visual_modal_value = pca_process(visual_modal_value, 128) # 128
            acoustic_modal_value = pca_process(acoustic_modal_value, 128) # 128     
            if not only_vat_concat:
                # 两两模态交叉
                va_hadamard = cross_array(visual_modal_value, acoustic_modal_value) # 192
                vt_hadamard = cross_array(visual_modal_value, textual_modal_value) # 73
                at_hadamard = cross_array(acoustic_modal_value, textual_modal_value) # 73       
                # 三个模态交叉
                vat_hadamard = cross_array_tri(visual_modal_value, acoustic_modal_value, textual_modal_value) # 73      
                # 两两模态拼接
                va_concat = concat_array(visual_modal_value, acoustic_modal_value) # 256
                vt_concat = concat_array(visual_modal_value, textual_modal_value) # 201
                at_concat = concat_array(acoustic_modal_value, textual_modal_value) # 201
            # 三个模态拼接
            vat_concat = concat_array_tri(visual_modal_value, acoustic_modal_value, textual_modal_value) # 329      
            if not only_vat_concat:
                # PCA降维
                va_concat = pca_process(va_concat, 128)
                vt_concat = pca_process(vt_concat, 128)
                at_concat = pca_process(at_concat, 128)
            vat_concat = pca_process(vat_concat, 128)       
            if not only_vat_concat:
                assert len(va_hadamard) == len(vt_hadamard) == len(at_hadamard) == len(vat_hadamard) == len(va_concat) == len(vt_concat) == len(at_concat) == len(vat_concat)       
            if not only_vat_concat:
                (va_hadamard_dict, vt_hadamard_dict, at_hadamard_dict, vat_hadamard_dict, 
                va_concat_dict, vt_concat_dict, at_concat_dict, vat_concat_dict) = ({}, {}, {}, {}, {}, {}, {}, {})
                for key, va_h, vt_h, at_h, vat_h, va_c, vt_c, at_c, vat_c, _ in zip(list(visual_modal_dict.keys()), 
                            va_hadamard, vt_hadamard, at_hadamard, vat_hadamard, 
                            va_concat, vt_concat, at_concat, vat_concat, tqdm(range(len(va_hadamard)))):
                    va_hadamard_dict[key] = np.array(va_h)
                    vt_hadamard_dict[key] = np.array(vt_h)
                    at_hadamard_dict[key] = np.array(at_h)
                    vat_hadamard_dict[key] = np.array(vat_h)
                    va_concat_dict[key] = np.array(va_c)
                    vt_concat_dict[key] = np.array(vt_c)
                    at_concat_dict[key] = np.array(at_c)
                    vat_concat_dict[key] = np.array(vat_c)
            else:
                vat_concat_dict = {}
                for key, vat_c, _ in zip(list(visual_modal_dict.keys()), vat_concat, tqdm(range(len(vat_concat)))):
                    vat_concat_dict[key] = np.array(vat_c)      
            if not only_vat_concat:
                tag_arr = double_modal_cross_arr+triple_modal_cross_arr+double_modal_concat_arr+triple_modal_concat_arr
                modal_arr = [va_hadamard_dict, vt_hadamard_dict, at_hadamard_dict, vat_hadamard_dict, 
                                        va_concat_dict, vt_concat_dict, at_concat_dict, vat_concat_dict]
            else:
                tag_arr = triple_modal_concat_arr
                modal_arr = [vat_concat_dict]
            for tag, modal_dict in zip(tag_arr, modal_arr): 
                save_matrix = []
                cnt, user_length = 0, len(user_feed)
                start = time.time()
                for userid in user_feed:
                    for feedid_click, _ in zip(user_feed[userid], tqdm(range(len(user_feed[userid])))):
                        modal_and_click = modal_dict[feedid_click[0]].tolist()+[feedid_click[1]]
                        if tag == 'va_cross':
                            arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'vt_cross':
                            arr = [userid]+visual_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'at_cross':
                            arr = [userid]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'vat_cross':
                            arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'va_concat':
                            arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'vt_concat':
                            arr = [userid]+visual_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'at_concat':
                            arr = [userid]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                        elif tag == 'vat_concat':
                            # 完全消融 即用单一userid替换embedding
                            if not user_embedding_exit:
                                arr = [userid]+[userid]+modal_and_click
                            # 不完全消融 即改变user embedding的维度
                            else:
                                arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                        else:
                            print(f'Error: matrix tag NOT exist!')
                            exit(1)
                        save_matrix.append(arr)
                    cnt += 1
                    print(f'{cnt} / {user_length} user has Processed. Each length of vector = {len(arr)}')
                end = time.time()
                print(f'{tag} matrix took {round((end-start)/60,3)} minutes.')      
                # 保存矩阵
                save_npy(path=os.path.join(matrix_path, dataset_tag+'_'+tag+'_matrix'), content=np.array(save_matrix))
            pass
        ########## tiktok ##########
        elif dataset_tag == tiktok:
            user_finish, user_like, user_device = get_userid_itemid(graph_path, tag=dataset_tag)
            for click_tag in [finish, like]:
                # 获取用户的embedding 
                visual_embedding_file = dataset_tag+'_'+click_tag+'_visual_user_embeddings.json'
                acoustic_embedding_file = dataset_tag+'_'+click_tag+'_acoustic_user_embeddings.json'
                textual_embedding_file = dataset_tag+'_'+click_tag+'_textual_user_embeddings.json'

                with rich.progress.open(os.path.join(user_embedding_path, visual_embedding_file), 'r') as f:
                    visual_userid_embedding = json.loads(f.read())
                with rich.progress.open(os.path.join(user_embedding_path, acoustic_embedding_file), 'r') as f:
                    acoustic_userid_embedding = json.loads(f.read())
                with rich.progress.open(os.path.join(user_embedding_path, textual_embedding_file), 'r') as f:
                    textual_userid_embedding = json.loads(f.read())

                # 获取模态信息
                visual_modal_dict = get_multimodal_dict(dataset_tag+'_visual')
                acoustic_modal_dict = get_multimodal_dict(dataset_tag+'_acoustic')
                textual_modal_dict = get_multimodal_dict(dataset_tag+'_textual')

                # 将模态全部归一化, 以免交叉时有影响
                visual_modal_value = list(visual_modal_dict.values())
                acoustic_modal_value = list(acoustic_modal_dict.values())
                textual_modal_value = list(textual_modal_dict.values())     
                visual_modal_value = MinMaxScaler().fit_transform(visual_modal_value) # 71
                acoustic_modal_value = MinMaxScaler().fit_transform(acoustic_modal_value) # 10 
                textual_modal_value = MinMaxScaler().fit_transform(textual_modal_value) # 22

                # 更新values
                assert len(visual_modal_dict) == len(visual_modal_value)
                assert len(acoustic_modal_dict) == len(acoustic_modal_value)
                assert len(textual_modal_dict) == len(textual_modal_value)
                for modal_dict,modal_value in zip([visual_modal_dict,acoustic_modal_dict,textual_modal_dict],
                                                    [visual_modal_value,acoustic_modal_value,textual_modal_value]):                
                    for (index, value),_ in zip(enumerate(list(modal_dict.keys())), tqdm(range(len(modal_value)))):
                        modal_dict[value] = modal_value[index]

                # 抖音数据集每个模态的keys对应不上
                common_itemid = list(visual_modal_dict.keys()&acoustic_modal_dict.keys()&textual_modal_dict.keys())
                print(f'There are {len(common_itemid)} in common among tree modals.') # 2420353
                
                # 只保留具有共同item_id的部分
                new_visual_dict, new_acoustic_dict, new_textual_dict = {}, {}, {}
                for key in common_itemid:
                    new_visual_dict[key] = visual_modal_dict[key]
                    new_acoustic_dict[key] = acoustic_modal_dict[key]
                    new_textual_dict[key] = textual_modal_dict[key]
                assert len(new_visual_dict)==len(new_acoustic_dict)==len(new_textual_dict)==len(common_itemid)
                visual_modal_dict = new_visual_dict
                acoustic_modal_dict = new_acoustic_dict
                textual_modal_dict = new_textual_dict
                visual_modal_value = list(new_visual_dict.values()) # 71
                acoustic_modal_value = list(new_acoustic_dict.values()) # 10
                textual_modal_value = list(new_textual_dict.values()) # 22

                if not only_vat_concat:
                    # 两两模态交叉
                    va_hadamard = cross_array(visual_modal_value, acoustic_modal_value) # 10
                    vt_hadamard = cross_array(visual_modal_value, textual_modal_value) # 22
                    at_hadamard = cross_array(acoustic_modal_value, textual_modal_value) # 10       
                    # 三个模态交叉
                    vat_hadamard = cross_array_tri(visual_modal_value, acoustic_modal_value, textual_modal_value) # 10      
                    # 两两模态拼接
                    va_concat = concat_array(visual_modal_value, acoustic_modal_value) # 81
                    vt_concat = concat_array(visual_modal_value, textual_modal_value) # 93
                    at_concat = concat_array(acoustic_modal_value, textual_modal_value) # 32       
                # 三个模态拼接
                vat_concat = concat_array_tri(visual_modal_value, acoustic_modal_value, textual_modal_value) # 103      

                if not only_vat_concat:
                    assert len(va_hadamard) == len(vt_hadamard) == len(at_hadamard) == len(vat_hadamard) == len(va_concat) == len(vt_concat) == len(at_concat) == len(vat_concat)       
                if not only_vat_concat:
                    (va_hadamard_dict, vt_hadamard_dict, at_hadamard_dict, vat_hadamard_dict, 
                    va_concat_dict, vt_concat_dict, at_concat_dict, vat_concat_dict) = ({}, {}, {}, {}, {}, {}, {}, {})
                    for key, va_h, vt_h, at_h, vat_h, va_c, vt_c, at_c, vat_c, _ in zip(list(visual_modal_dict.keys()), 
                                va_hadamard, vt_hadamard, at_hadamard, vat_hadamard, 
                                va_concat, vt_concat, at_concat, vat_concat, tqdm(range(len(va_hadamard)))):
                        va_hadamard_dict[key] = np.array(va_h)
                        vt_hadamard_dict[key] = np.array(vt_h)
                        at_hadamard_dict[key] = np.array(at_h)
                        vat_hadamard_dict[key] = np.array(vat_h)
                        va_concat_dict[key] = np.array(va_c)
                        vt_concat_dict[key] = np.array(vt_c)
                        at_concat_dict[key] = np.array(at_c)
                        vat_concat_dict[key] = np.array(vat_c)
                else:
                    vat_concat_dict = {}
                    for key, vat_c, _ in zip(list(visual_modal_dict.keys()), vat_concat, tqdm(range(len(vat_concat)))):
                        vat_concat_dict[key] = np.array(vat_c)      
                if not only_vat_concat:
                    tag_arr = double_modal_cross_arr+triple_modal_cross_arr+double_modal_concat_arr+triple_modal_concat_arr
                    modal_arr = [va_hadamard_dict, vt_hadamard_dict, at_hadamard_dict, vat_hadamard_dict, 
                                            va_concat_dict, vt_concat_dict, at_concat_dict, vat_concat_dict]
                else:
                    tag_arr = triple_modal_concat_arr
                    modal_arr = [vat_concat_dict]
                for tag, modal_dict in zip(tag_arr, modal_arr): 
                    if click_tag == finish:
                        user_feed = user_finish
                    elif click_tag == like:
                        user_feed = user_like

                    save_matrix = []
                    cnt, user_length = 0, len(user_feed)
                    start = time.time()
                    for userid in user_feed:
                        for feedid_click, _ in zip(user_feed[userid], tqdm(range(len(user_feed[userid])))):
                            if feedid_click[0] in modal_dict:
                                modal_and_click = modal_dict[feedid_click[0]].tolist()+[feedid_click[1]]
                            else:
                                continue
                            if tag == 'va_cross':
                                arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'vt_cross':
                                arr = [userid]+visual_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'at_cross':
                                arr = [userid]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'vat_cross':
                                arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'va_concat':
                                arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'vt_concat':
                                arr = [userid]+visual_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'at_concat':
                                arr = [userid]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                            elif tag == 'vat_concat':
                                # 完全消融 即用单一userid替换embedding
                                if not user_embedding_exit:
                                    arr = [userid]+[userid]+modal_and_click
                                # 不完全消融 即改变user embedding的维度
                                else:
                                    arr = [userid]+visual_userid_embedding[str(userid)]+acoustic_userid_embedding[str(userid)]+textual_userid_embedding[str(userid)]+modal_and_click
                            else:
                                print(f'Error: matrix tag NOT exist!')
                                exit(1)
                            save_matrix.append(arr)
                        cnt += 1
                        print(f'{cnt} / {user_length} user has Processed. Each length of vector = {len(arr)}')
                    end = time.time()
                    print(f'{tag} matrix took {round((end-start)/60,3)} minutes.')      
                    # 保存矩阵
                    save_npy(path=os.path.join(matrix_path, dataset_tag+'_'+click_tag+'_'+tag+'_matrix'), content=np.array(save_matrix))
        else:
            print(f'Error: wrong dataset tag = {dataset_tag}. dataset tag should be {wechat} or {tiktok}')
    exit(0)