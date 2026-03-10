import h5py
from atm.dataloader import ATMPretrainDataset


def read_hdf5_file(file_path):
    try:
        # 以只读模式 ('r') 打开文件
        with h5py.File(file_path, 'r') as f:
            
            # 1. 查看文件中所有的主键（根目录下的组或数据集）
            print(f"文件结构 (Keys): {list(f.keys())}")
            
            # 2. 递归遍历并打印文件内所有对象的名字
            print("\n文件详细树状结构:")
            f.visit(lambda name: print(f" - {name}"))

            pass

    except Exception as e:
        print(f"读取文件时出错: {e}")


def test_read():
    read_hdf5_file('/data2/jibaixu/Datasets/LIBERO/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5')


def test_dataset():
    cfg = {
        "train_dataset": ['./data/atm_libero/libero_object/*/train/'],
        "dataset_cfg": {'img_size': 128, 'frame_stack': 1, 'num_track_ts': 16, 'num_track_ids': 32, 'cache_all': True, 'cache_image': False},
        "aug_prob": 0.9,
    }
    train_dataset = ATMPretrainDataset(dataset_dir=cfg["train_dataset"], **cfg["dataset_cfg"], aug_prob=cfg["aug_prob"])
    print(f"{train_dataset[0]}")

test_dataset()
