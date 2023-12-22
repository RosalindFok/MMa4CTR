import os
import orjson as json

# 加载路径
path_file = os.path.join('path.json')
if os.path.exists(path_file):
    with open(path_file, 'r') as f:
        path = json.loads(f.read())
else:
    print(f'{path_file} NOT exists!')

# 文件路径
parent_folder = path['parent_folder']
graph_path = os.path.join(parent_folder, path['graph_path'])
modal_path = os.path.join(parent_folder, path['modal_path'])
matrix_path = os.path.join(parent_folder, path['matrix_path'])
wechat_dataset_path = os.path.join(parent_folder, path['dataset_path'], path['wechat_dataset_path'])
tiktok_dataset_path = os.path.join(parent_folder, path['dataset_path'], path['tiktok_dataset_path'])
torch_model_path = os.path.join(parent_folder, path['torch_model_path'])
user_embedding_path = os.path.join(parent_folder, path['user_embedding_path'])
saved_path = os.path.join('.', path['saved_path'])
temp_path = os.path.join(parent_folder, path['temp_path'])

# 模态拼接、交叉的标签名
each_modal_arr = ['visual', 'acoustic', 'textual']
double_modal_cross_arr = ['va_cross', 'vt_cross', 'at_cross']
triple_modal_cross_arr = ['vat_cross']
double_modal_concat_arr = ['va_concat', 'vt_concat', 'at_concat']
triple_modal_concat_arr = ['vat_concat']

# 数据集
tiktok = 'tiktok'.lower()
wechat = 'wechat'.lower()
# 抖音数据集的两个点击
finish = 'finish'
like = 'like'