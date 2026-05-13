
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models import Detections
import math
import itertools
import time
import os
from os.path import normpath, join, isdir, exists

base_folder = './'


folder_names = [p for p in os.listdir(base_folder) if isdir(p)]
folder_names.sort()
model_paths = [normpath(join(base_folder, p, "model", "network.rpk")) for p in folder_names]


seconds = 30
framerate = 30
measurements = framerate * seconds + 2
print(folder_names)

fps_folder = '../networks_fps'


class Model_but_not_abstract(Model):
    """Generic model"""

    def __init__(self, model_path):
        """Initialize the YOLO model for IMX500 deployment."""
        super().__init__(
            model_file=model_path,  # replace with proper directory
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
            
        )

    def post_process(self, output_tensors):
        """No need for pp"""
        
        return output_tensors

start = 0
end = 0

for model_path in model_paths:
	
	model_name = model_path.split('/')[0]
	save_path = normpath(join(fps_folder, model_name))
	if exists(save_path):
		print(f"Skipping {model_name} because fps file already exists")
		continue
	
	print("Initializing camera and model")
	
	resolution = [int(n) for n in model_path.split('_')[1:3]]
	
	model_latencies = []
	
	print(f"\nWorking with model {model_path} at resolution {resolution}\n")
	
	#print(resolution)
	# input()
	
	device = AiCamera(frame_rate=framerate, image_size=resolution)  
	model = Model_but_not_abstract(model_path)
            
	device.deploy(model)

	i = 0

	with device as stream:
		
		# print("The stream starts here")
		
		
		for frame in stream:
			
			end = time.time()
			model_latencies.append(end - start)
			
			if i > measurements:
				break
			
			# print("The frames starts here")
			
			
			
			pass
			# frame.display()
			
			i += 1
			start = time.time()
			
			print(f"{i} / {measurements} measurements", end = '\r')
			
			
	model_latencies = model_latencies[2:]
	
	# print(model_latencies)
	
	avg_lat = sum(model_latencies) / len(model_latencies)
	
	print(f"Average demo latency: {avg_lat * 1000} [ms] ({1 / avg_lat} fps")
	
	
	print("Press enter to continue")
	input()
	
	print(f"Saving result to {fps_folder}")
	
	with open(save_path, 'w') as f:
		f.write(f"{1 / avg_lat}")
	
	

