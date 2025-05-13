import os
import subprocess
import json
import shutil
from pathlib import Path


INPUT_FOLDER = r"temp_video"  
OUTPUT_FOLDER = r"temp_video(30s)"  
SEGMENT_DURATION = 30  # 视频片段时长（秒）
MIN_DURATION = 30  # 需要处理的最小视频时长（秒）

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams', 
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = json.loads(result.stdout)
    
    # 尝试从不同位置获取持续时间
    if 'format' in data and 'duration' in data['format']:
        return float(data['format']['duration'])
    
    for stream in data.get('streams', []):
        if 'duration' in stream:
            return float(stream['duration'])
    
    return 0

def split_video(video_path, output_folder, segment_duration=30):
    """将视频分割成指定时长的片段"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_pattern = os.path.join(output_folder, f"{video_name}_%03d.mp4")
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-c', 'copy',  # 使用相同的编解码器（不重新编码）
        '-map', '0',   # 包含所有流（视频、音频等）
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-reset_timestamps', '1',
        '-avoid_negative_ts', 'make_zero',
        output_pattern
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    

def copy_video(video_path, output_folder):
    """复制视频到输出文件夹"""
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_folder, video_name)
    
    shutil.copy2(video_path, output_path)
    print(f"Copied video: {video_path} -> {output_path}")

def process_folder(input_folder, output_folder, segment_duration, min_duration):
    """处理文件夹中的所有视频文件"""
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的视频扩展名
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    
    # 获取文件夹中的所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(input_folder).glob(f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # 处理每个视频文件
    for video_path in video_files:
        video_path_str = str(video_path)
        duration = get_video_duration(video_path_str)
        
        if duration >= min_duration:
            print(f"Processing video: {video_path_str} (Duration: {duration:.2f} seconds)")
            split_video(video_path_str, output_folder, segment_duration)
        else:
            print(f"Video duration is less than {min_duration} seconds, copying directly: {video_path_str} (Duration: {duration:.2f} seconds)")
            copy_video(video_path_str, output_folder)

if __name__ == "__main__":
    print(f"Starting video processing...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Video segment duration: {SEGMENT_DURATION} seconds")
    print(f"Minimum video duration to process: {MIN_DURATION} seconds")
        
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, SEGMENT_DURATION, MIN_DURATION)
    print("All videos processed!")
