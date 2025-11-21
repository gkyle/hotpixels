import os

current_directory = os.getcwd()
dot_path = os.path.join(current_directory, 'hotpixels.lrdevplugin/PluginPath.txt')
lrc_path = os.path.join(current_directory, "run.bat")

with open(dot_path, 'w') as file:
    file.write(lrc_path)
    file.close()
