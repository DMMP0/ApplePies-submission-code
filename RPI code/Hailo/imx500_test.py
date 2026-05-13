

import numpy as np
from hailo_inference import InferPipeline
import math
import itertools
import picamera2
import time
from os.path import normpath, join, exists
from os import listdir, makedirs
from picamera2 import Picamera2


print("imported ok")


# exit(0)

# list of models

models_folder = './models/done'
check_folder = "./flags"
makedirs(check_folder, exist_ok=True)
model_names = listdir(models_folder)
model_names.sort()
models_abs_paths = [normpath(join(models_folder, p)) for p in model_names]

fps = 30
warmup_seconds = 8 # seconds the accelerator needs tostabilize the power consumption, empirical value found with repeated measurements
seconds = 30 # NB: inf + fps roughly = 1s, but it's not exact
# delay = 0.01

tot_frames = fps * (seconds + warmup_seconds)

# print(models_abs_paths)

print('Starting measurements')

camera = Picamera2()
camera.video_configuration.controls.FrameRate = fps
camera.video_configuration.main.format = "RGB888"

for i, model_path in enumerate(models_abs_paths):
	
	#			remove .hef
	l = model_names[i][:-4].split('_')
	res = (int(l[-2]), int(l[-1]))
	model_name = '_'.join(l[:-2])
	
	flag_file_path = normpath(join(check_folder, model_names[i]))
	if exists(flag_file_path):
		print(f"Skipping {model_name} at {res} because {flag_file_path} exists")
		continue
	
	#print(res)
	print(f"\nWorking with {model_name} at resolution {res}\n")
	
	print(f"\nLoading {model_path}\n")
	
	# setup hailo model

	infer_pipeline = InferPipeline(net_path=model_path, batch_size=1)
	
	print(f"\n{model_path} loaded \n")
	
	print(f"\nInitializing camera\n")

	camera.video_configuration.main.size = res

	# print(camera.video_configuration)
	# exit(0)

	camera.start("video")
	
	print(f"\nStarting in")
	print("3")
	time.sleep(1)
	print("2")
	time.sleep(1)
	print("1")
	time.sleep(1)
	
	for f in range(tot_frames):
		print(f"Frame {f} / {tot_frames}", end='\r')
		array = camera.capture_array("main")

		# print(array.shape)
		result = infer_pipeline.infer_pipeline([array])
		# print(result)
	
		# exit(0)
		# no need for the sleep
	camera.stop()
	infer_pipeline.close()
	
	print("Inference done, press enter to continue")
	input()
	
	# save flag for repeated executions
	
	with open(flag_file_path, 'w') as f:
		f.write("done")
	
	# exit(0)

exit(0)


        
