import os
import subprocess
import shutil
from pathlib import Path

# 检查是否安装了ImageMagick
try:
    subprocess.run(['magick', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
except FileNotFoundError:
    print("ImageMagick未安装，请前往 https://imagemagick.org/ 下载安装。")
    exit()

# 转换目录下的HEIC文件为JPG
target_directory = 'data/手语照片'  # 替换为你的目标目录路径

output_directory = 'data/img'

img_files = [f for f in os.listdir(target_directory)]

txt = os.path.join(Path(output_directory).parent, 'data.txt')
txt_file = open(txt, mode='w', encoding='utf-8')
i = 0
for file in img_files:
    i += 1
    input_file = os.path.join(target_directory, file)
    filename = os.path.splitext(input_file)[0]
    name = os.path.basename(filename)
    suffix = os.path.splitext(input_file)[1]
    if suffix == '.heic':
        output_file = os.path.join(output_directory, f'{str(i).rjust(5, "0")}.jpg')
        try:
            subprocess.run(['magick', input_file, output_file], check=True)
            print(f"{file} 转换完成为 {os.path.basename(output_file)}")
            txt_file.write(f'{name}|{output_file}\n')
        except subprocess.CalledProcessError as e:
            print(f"转换 {file} 失败:", e)
    else:
        output_file = os.path.join(output_directory, f'{str(i).rjust(5, "0")}{suffix}')
        shutil.copy(input_file, output_file)
        txt_file.write(f'{name}|{output_file}\n')
txt_file.close()

print("转换完成。")
