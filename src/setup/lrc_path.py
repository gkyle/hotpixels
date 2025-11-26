import os

current_directory = os.getcwd()
dot_path = os.path.join(current_directory, 'hotpixels.lrdevplugin/PluginPath.txt')

# Determine if OS is Windows
if os.name == 'nt':
    lrc_path = os.path.join(current_directory, "run.bat")
else:
    lrc_path = os.path.join(current_directory, "run.sh")

with open(dot_path, 'w') as file:
    file.write(lrc_path)
    file.close()
