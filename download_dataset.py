# 1. 验证 Token (保持你原来的代码)
from modelscope.hub.api import HubApi
api = HubApi()
api.login('ms-ae02ff7d-db79-48c3-b0b4-36967e8cdbdf')

# 2. 使用 snapshot_download 下载到指定目录
from modelscope.hub.snapshot_download import snapshot_download

# 在这里指定你想要的路径
your_local_path = '/data_jbx/Datasets/Realbot'

snapshot_download(
    'jibaixu/Realbot0407', 
    repo_type='dataset', 
    local_dir=your_local_path
)

print(f"数据集已成功下载到: {your_local_path}")
