import matlab.engine
import numpy as np
import os
from plot_func import *

try:
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()

    root_path = r'D:\Documents\Self_Files\ForwardBackward Process\projects\正反演代码整理版_96 - 已修正\正反演代码整理版_96 - 已修正'
    eng.addpath(eng.genpath(root_path), nargout=0)
    eng.cd(root_path, nargout=0)

    Param = eng.Param4python()

    source_dir = 'D:\Documents\Self_Files\Projects\SceneGenerating2\output_coastline_patches\\npy_files'
    # source_dir = 'D:\Documents\Self_Files\Projects\SceneGenerating\data\data17Mars\Binary'
    output_dir = 'data\\data6Mai'
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]

    # 按序号排序（提取数字部分）
    def extract_number(filename):
        # 从scene_XXXX.npy中提取XXXX
        try:
            return int(filename.split('_')[1].split('.')[0])
        except:
            return -1

    npy_files.sort(key=extract_number)

    # 设置划分比例
    train_ratio = 0.95  # 80% 训练集，20% 测试集

    # 使用 numpy 随机划分
    np.random.seed(42)  # 设置随机种子，确保结果可重复
    indices = np.random.permutation(len(npy_files))
    train_size = int(len(npy_files) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_files = [npy_files[i] for i in train_indices]
    test_files = [npy_files[i] for i in test_indices]

    print("处理训练集...")
    antenna_position = np.array(Param['Array']['ant_pos'], dtype=np.double)
    np.save(os.path.join(train_dir, f"array_position.npy"), antenna_position)

    for i, filename in enumerate(train_files, 1):
        print(f"\r处理第 {i}/{len(train_files)} 个文件: {filename}", end="")

        # 读取原始数据
        filepath = os.path.join(source_dir, filename)
        original_data = np.load(filepath)

        # 乘以90
        scene_TB = original_data
        scene_TB = np.array(scene_TB, dtype=np.double)

        visibility, observe_TB = eng.TB4python_V2(Param, scene_TB, nargout=2)
        observe_TB = np.array(observe_TB, dtype=np.double)
        visibility = np.array(visibility, dtype=np.double)
        # res = np.array(res, dtype=np.complex128)
        # R = np.array(R, dtype=np.complex128)

        # 提取序号（保持4位数字）
        try:
            raise Exception("手动触发异常，跳转到 except")
            # 从scene_XXXX.npy中提取XXXX
            num_str = filename.split('_')[1].split('.')[0]
            # 生成新文件名：observe_XXXX.npy
            scene_filename = f"scene_{num_str}.npy"
            observe_filename = f"observe_{num_str}.npy"
            visibility_filename = f"visibility_{num_str}.npy"
            observe_fig = f"observe_{num_str}.png"
        except:
            # 如果文件名格式不符，使用原始序号
            scene_filename = f"scene_{i:04d}.npy"
            observe_filename = f"observe_{i:04d}.npy"
            visibility_filename = f"visibility_{i:04d}.npy"
            observe_fig = f"observe_{i:04d}.png"

        save_comparison_png(scene_TB, observe_TB, output_path=os.path.join(train_dir, observe_fig))
        # 保存处理后的数据
        scene_output_path = os.path.join(train_dir, scene_filename)
        observe_output_path = os.path.join(train_dir, observe_filename)
        visibility_output_path = os.path.join(train_dir, visibility_filename)
        np.save(scene_output_path, scene_TB)
        np.save(observe_output_path, observe_TB)
        np.save(visibility_output_path, visibility)

    print("\n")

    # 处理测试集
    print("处理测试集...")
    np.save(os.path.join(test_dir, f"array_position.npy"), antenna_position)

    for i, filename in enumerate(test_files, 1):
        print(f"\r处理测试集第 {i}/{len(test_files)} 个文件: {filename}", end="")

        # 读取原始数据
        filepath = os.path.join(source_dir, filename)
        original_data = np.load(filepath)

        scene_TB = original_data
        scene_TB = np.array(scene_TB, dtype=np.double)

        visibility, observe_TB = eng.TB4python_V2(Param, scene_TB, nargout=2)
        observe_TB = np.array(observe_TB, dtype=np.double)
        visibility = np.array(visibility, dtype=np.double)

        try:
            raise Exception("手动触发异常，跳转到 except")
            num_str = filename.split('_')[1].split('.')[0]
            scene_filename = f"scene_{num_str}.npy"
            observe_filename = f"observe_{num_str}.npy"
            visibility_filename = f"visibility_{num_str}.npy"
            observe_fig = f"observe_{num_str}.png"
        except:
            scene_filename = f"scene_{i:04d}.npy"
            observe_filename = f"observe_{i:04d}.npy"
            visibility_filename = f"visibility_{i:04d}.npy"
            observe_fig = f"observe_{i:04d}.png"

        save_comparison_png(scene_TB, observe_TB, output_path=os.path.join(test_dir, observe_fig))
        scene_output_path = os.path.join(test_dir, scene_filename)
        observe_output_path = os.path.join(test_dir, observe_filename)
        visibility_output_path = os.path.join(test_dir, visibility_filename)
        np.save(scene_output_path, scene_TB)
        np.save(observe_output_path, observe_TB)
        np.save(visibility_output_path, visibility)

    print("\n")

    # 关闭 MATLAB 引擎
    eng.quit()

except Exception as e:
    print("调用 MATLAB 出错:", e)

# print(test)