import os

currentDirectory = os.getcwd()
dotPath = os.path.join(currentDirectory, 'hotpixels.lrdevplugin/PluginPath.txt')

# Determine if OS is Windows
if os.name == 'nt':
    lrcPath = os.path.join(currentDirectory, "run.bat")
else:
    lrcPath = os.path.join(currentDirectory, "run.sh")

with open(dotPath, 'w') as file:
    file.write(lrcPath)
    file.close()
