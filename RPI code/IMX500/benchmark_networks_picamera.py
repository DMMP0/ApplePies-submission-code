
from picamera2.devices import IMX500
from picamera2 import CompletedRequest, MappedArray, Picamera2
import time
import os
from os.path import normpath, join, isdir

base_folder = './'


folder_names = [p for p in os.listdir(base_folder) if isdir(p)]
folder_names.sort()
model_paths = [normpath(join(base_folder, p, "model", "network.rpk")) for p in folder_names]

framerate = 30
measurements = 900
sleep_pause = 0.05
benchmark_energy = True
print(folder_names)


for i in range(len(folder_names)):
	
	folder_name = folder_names[i]
	network_path = model_paths[i]
	
	if folder_name[0] == ".":
		print(f"Skipping {folder_name}, as it was already benchmarked")
		continue
	   
	   
	print(f'Starting latency measurement of {folder_name}')
	imx500 = IMX500(network_path)
	picam2 = Picamera2(imx500.camera_num)
	config = picam2.create_preview_configuration(controls={"FrameRate": framerate}, buffer_count=12)

	imx500.show_network_fw_progress_bar()
	imx500.set_auto_aspect_ratio()
	
	
	picam2.start(config, show_preview=False)
	if not benchmark_energy:
		latencies_dsp = []
		latencies_inf = []
	
	for i in range(measurements):
		
		request = picam2.capture_request()
		metadata = request.get_metadata()
		if metadata:
			# print(metadata)
			if not benchmark_energy:
				inf_time, dsp_info = metadata['CnnKpiInfo']
				
				tot = inf_time + dsp_info
				
				if tot != 0:
					latencies_dsp.append(dsp_info)
					latencies_inf.append(inf_time)
			else:
				pass
		request.release()
		
		# if tot != 0:
			# print(f"{inf_time}, {dsp_info} = {inf_time + dsp_info}")
		print(f"{i} / {measurements} measurement", end = '\r')
		
		if benchmark_energy:
			time.sleep(sleep_pause)
			pass
	picam2.close()
	
	if not benchmark_energy:
		avg_dsp = sum(latencies_dsp) / len(latencies_dsp)
		avg_inf = sum(latencies_inf) / len(latencies_inf)
		
		
		print(f"\tAverage dsp latency: {avg_dsp}")
		print(f"\tAverage inf latency: {avg_inf}")
	else:
		print("\n\nPress to continue\n\n")
	
		input()
