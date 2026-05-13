import os
import shutil
from os.path import normpath, join, isdir


base_folder = './'

packer_out_folders = [normpath(join(base_folder, p)) for p in os.listdir(base_folder) if isdir(p)]

# print(packer_out_folders)

for folder in packer_out_folders:
	
	print("\nWorking with " + folder)
	packer_out_file = normpath(join(folder, "packerOut.zip"))
	model_folder = normpath(join(folder, "model"))
	command = "imx500-package -i " + packer_out_file + " -o " + model_folder
	os.system(command)
