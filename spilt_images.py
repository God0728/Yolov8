import os
import random
import shutil

def split_images(source_dir, output_dir1, output_dir2, ratio=0.9):
    """
    将指定路径下的 .jpg 文件按照比例随机分成两个文件夹。

    参数:
        source_dir (str): 源文件夹路径，包含 .jpg 文件。
        output_dir1 (str): 第一个目标文件夹路径，存放比例较大的文件。
        output_dir2 (str): 第二个目标文件夹路径，存放比例较小的文件。
        ratio (float): 分配比例，默认为 0.9（即 90% 的文件放入 output_dir1）。
    """
    # 确保目标文件夹存在
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # 获取所有 .jpg 文件
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)  # 随机打乱文件列表

    # 计算分割点
    split_index = int(len(image_files) * ratio)

    # 将文件分配到两个文件夹
    for i, file_name in enumerate(image_files):
        src_path = os.path.join(source_dir, file_name)
        if i < split_index:
            dst_path = os.path.join(output_dir1, file_name)
        else:
            dst_path = os.path.join(output_dir2, file_name)
        shutil.move(src_path, dst_path)  # 移动文件

    print(f"文件已成功分割：{len(image_files[:split_index])} 个文件放入 {output_dir1}，{len(image_files[split_index:])} 个文件放入 {output_dir2}")

# 示例用法
if __name__ == "__main__":
    source_dir = "/root/autodl-tmp/bizhang0223/images/"  # 替换为你的源文件夹路径
    output_dir1 = "/root/autodl-tmp/bizhang0223/train"  # 替换为第一个目标文件夹路径
    output_dir2 = "/root/autodl-tmp/bizhang0223/val"  # 替换为第二个目标文件夹路径

    split_images(source_dir, output_dir1, output_dir2)