import os 
import json
import random
dfdcroot='/mnt/data/wxp/cropped_dfdc'
dfwroot='/mnt/data/wxp/cropped_dfw'
celebroot='/mnt/data/wxp/cropped_celeb'
ffpproot='/mnt/data/wxp/FaceForensics++'
dfdcppproot='/mnt/data/wxp/cropped_dfdcp'
def load_json(name):
    try:
        with open(name) as f:
            a=json.load(f)
            return a
    except FileNotFoundError:
        print(f"File {name} not found.")


def catdir(dir,label):
    l=os.listdir(dir)
    return [[os.path.join(dir,i),label] for i in l]

def FF_dataset(tag='Origin',codec='c23',part='train', split_ratio=(0.7, 0.15, 0.15)):
    assert(tag in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face','FaceShifter'])
    assert(codec in ['c23','c40','all'])
    assert(part in ['train','val','test','all'])
    train_ratio, val_ratio, test_ratio = split_ratio
    assert train_ratio + val_ratio + test_ratio == 1
    # codec='c23'
    if part=="all":
        return FF_dataset(tag,codec,'train', split_ratio)+FF_dataset(tag,codec,'val', split_ratio)+FF_dataset(tag,codec,'test', split_ratio)
    path = os.path.join(ffpproot, codec,part)
    json_path = os.path.join(ffpproot, 'splits', f'{part}.json')
    metafile=load_json(json_path)
    files=[]
    if metafile is None:
        raise ValueError(f"metafile is None! Check dataset path for {path }")

    if tag=='Origin':
        path = os.path.join(path, 'real')
        for i in metafile:
            # print(i)
            file_path = os.path.join(path, i[1])
            files.append([file_path,0])
            # print('origin')
    else:
        path=os.path.join(path, 'fake',tag,)
        for i in metafile:
            # print(i)
            file_path1 = os.path.join(path, f"{i[0]}_{i[1]}")
            files.append([file_path1,1])
            file_path2 = os.path.join(path, f"{i[1]}_{i[0]}")
            files.append([file_path2,1])
            # print('这里为啥全为1')

    for f in files[:1]:
        print(f)
    return files
def celeb_dataset(part='train'):
    celeb_path ='/mnt/data/wxp/cropped_celeb/List_of_testing_videos.txt'#/home/wxp
    with open(celeb_path, 'r') as f:
        celeb_data = []
        for line in f:
            parts = line.strip().split(maxsplit=1)  # 关键修改：限制分割次数
            if len(parts) >= 2:
                label_str, file_name = parts[0], parts[1]  # 调换顺序
                file_name = file_name.replace(".mp4", "")  # 新增此行
                try:
                    label = int(label_str)
                    celeb_data.append([file_name, label])    # 存储顺序为 [文件名, 标签]
                except ValueError:
                    print(f"忽略无效行：{line.strip()}（标签 '{label_str}' 不是整数）")
    samples = []
    more_than_30_frames = 0  # 用于统计超过30帧的视频数量
    for video_name, label in celeb_data:
        video_dir = os.path.join(celebroot, video_name)
        if not os.path.isdir(video_dir):
            continue
        frames = sorted(os.listdir(video_dir))
        # 按间隔采样
        if len(frames) > 30:
            more_than_30_frames += 1
            frames = frames[::10]  # 按间隔采样
            frames = frames[:30]  # 限制最多30帧
            for frame_name in frames:
                frame_path = os.path.join(video_dir, frame_name)
                if os.path.isfile(frame_path):
                    samples.append([frame_path, 1 - label])  # 标签反转

    for f in samples[:1]:
        print(f)
    return samples
def dfdc_dataset(part='test'):
    assert(part in ['test'])
    lf=load_json('/mnt/data/wxp/cropped_dfdc/metadata.json')
    files = []
    more_than_30_frames = 0  # 用于统计超过30帧的视频数量
    for fname, info in lf.items():
        name, _ = os.path.splitext(fname)  # 去除.mp4
        full_path = os.path.join(dfdcroot,  name)  # 假设帧图像文件夹与视频名一致
        if not os.path.isdir(full_path):
            # 目录不存在，直接跳过
            # print(f'目录不存在跳过{full_path}')
            continue
        frames = sorted([f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(frames) == 0:
            continue
        max_frames=30
        interval=10
        if len(frames) > max_frames:
            interval = len(frames) // max_frames
            frames = frames[::interval][:max_frames]
            more_than_30_frames += 1  # 统计超过30帧的视频
        else:
            frames = frames[:max_frames]
        label = info.get('is_fake', 1)
        for fn in frames:
            files.append([os.path.join(full_path, fn), label])
        # files.append([full_path, label])
    # print(f"超过30帧的视频个数: {more_than_30_frames}")
    # 输出前几个检查
    for f in files[:1]:
        print(f)
    print(f"DFDC  samples: {len(files)}")

    return files
def dfw_dataset(part='test', root_frames="/mnt/data/wxp/cropped_dfw"):
    assert part in ('test')
    samples = []
    # base = os.path.join(root_frames, part)
    base = root_frames
    for cls_name, label in [('real',0), ('fake',1)]:
        cls_root = os.path.join(base, cls_name)
        if not os.path.isdir(cls_root):
            continue
        more_than_30_frames = 0  # 用于统计超过30帧的视频数量
        for video_id in os.listdir(cls_root):
            video_path = os.path.join(cls_root, video_id)
            if not os.path.isdir(video_path):
                continue
            # 遍历最内层的帧序列目录
            for seq in os.listdir(video_path):
                seq_path = os.path.join(video_path, seq)
                if not os.path.isdir(seq_path):
                    continue  # 不是文件夹就跳过

                # 获取这个文件夹里的帧
                frames = sorted(os.listdir(seq_path))
                n_frames = 30  # 你想采样30帧
                if len(frames) < n_frames:
                    sampled_frames = frames
                else:
                    interval = len(frames) // n_frames
                    sampled_frames = frames[::interval][:n_frames]
                    more_than_30_frames += 1  # 统计超过30帧的视频
                # 把路径列表存进 samples
                for f in sampled_frames:
                    samples.append([os.path.join(seq_path, f), label])

        # print(f"DFW ({part}) samples: {len(samples)}")
        # print(f"超过30帧的视频个数: {more_than_30_frames}")
    for f in samples[:1]:
        print(f)
    return samples
import os
import json
from tqdm import tqdm
def celebv1_dataset(json_path="/mnt/data/wxp/cropped_celebv1/celebv1_cropped.json", part='test', max_frames=30):
    """
    解析 Celeb-DF-v1 数据集 JSON
    仅加载 test 部分，每个视频最多保留 30 帧
    label: 0=real, 1=fake
    """
    if not os.path.isfile(json_path):
        print(f"⚠️ 找不到 JSON 文件: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    samples = []
    more_than_30 = 0
    debug_samples = []

    # print(f"\n📄 加载 CelebV1 JSON: {json_path}, part={part}")

    celeb_root = data_json.get("Celeb-DF-v1", {})
    for split_name, split_content in celeb_root.items():  # CelebDFv1_real, CelebDFv1_fake
        label = 0 if "real" in split_name.lower() else 1
        # print(f"  split: {split_name}, label={label}")

        for part_name, videos in split_content.items():
            if part_name.lower() != part.lower():
                continue
            # print(f"    处理 part: {part_name}")

            for video_id, video_info in tqdm(videos.items(), desc=f"CelebV1-{split_name}-{part_name}"):
                frames = video_info.get("frames", [])
                if not frames:
                    continue

                if len(frames) > max_frames:
                    more_than_30 += 1
                    step = max(1, len(frames) // max_frames)
                    frames = frames[::step][:max_frames]

                for f in frames:
                    if os.path.isfile(f):
                        samples.append([f, label]) 

    # print(f"\n✅ CelebV1 {part} 样本数: {len(samples)}")
    # print(f"超过{max_frames}帧的视频数: {more_than_30}")
    print(f"前几个样本示例: {samples[:1]}")

    return samples
