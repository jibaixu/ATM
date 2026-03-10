import argparse
import os
import glob
from pathlib import Path


def merge_libero_datasets(source_folder, target_folder):
    """
    将各个 suite/task 中的 train 和 val 数据合并到统一的目录下。
    """
    # 定义全局合并后的路径
    global_train_dir = os.path.join(target_folder, 'train')
    global_val_dir = os.path.join(target_folder, 'val')

    os.makedirs(global_train_dir, exist_ok=True)
    os.makedirs(global_val_dir, exist_ok=True)

    print(f"开始从 {source_folder} 合并数据到 {target_folder}...")

    # 遍历 suite (例如: libero_spatial, libero_10 等)
    for suite_name in os.listdir(source_folder):
        suite_path = os.path.join(source_folder, suite_name)
        if not os.path.isdir(suite_path):
            continue

        # 遍历 task
        for task_name in os.listdir(suite_path):
            task_path = os.path.join(suite_path, task_name)
            if not os.path.isdir(task_path):
                continue

            # 处理 train 和 val 子文件夹
            for split in ['train', 'val']:
                src_split_dir = os.path.join(task_path, split)
                dst_split_dir = global_train_dir if split == 'train' else global_val_dir

                if os.path.exists(src_split_dir):
                    files = glob.glob(os.path.join(src_split_dir, '*.hdf5'))
                    
                    for f in files:
                        filename = os.path.basename(f)
                        # 为了避免冲突，新文件名为: suite_task_original-name.hdf5
                        new_name = f"{suite_name}_{task_name}_{filename}"
                        dst_path = os.path.join(dst_split_dir, new_name)

                        # 如果软链接已存在，先删除
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        
                        # 创建绝对路径的软链接，确保移动脚本后依然有效
                        abs_src = os.path.abspath(f)
                        os.symlink(abs_src, dst_path)
    
    print(f"合并完成！")
    print(f"训练集总量: {len(os.listdir(global_train_dir))}")
    print(f"验证集总量: {len(os.listdir(global_val_dir))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge LIBERO train/val splits from all suites.")
    parser.add_argument('--folder', type=str, default='./data/atm_libero/', help="原始预处理后的数据根目录")
    parser.add_argument('--output', type=str, default='./data/atm_libero_merged/', help="合并后的数据存储目录")
    args = parser.parse_args()

    merge_libero_datasets(args.folder, args.output)
