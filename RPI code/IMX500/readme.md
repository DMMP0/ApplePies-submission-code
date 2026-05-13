# Requirements
This scripts require two specific libraries to be installed in the RPI environment:

- modlib:
  - https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library
  - https://pypi.org/project/modlib/
- imx500 Packager
  - https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-packager?version=2025-09-30&progLang=

# Structure

During the measurements, we had one folder for each combination of network and resolution.

The file create_rpk.py iterates on the current folder and creates the .rpk (IMX500 firmware) file for each network.

Since the measuring sessions can be long, we set the '.' character at the beginning of the folder as a skipping mark for already performed measurements (to be set manually).
As such, the file 'reset_folder_names.py' simply removes that character.

# Latency measurements

At the time of writing, proper latency measurements can only be performed with picamera2.
Therefore, the file 'benchmark_networks_picamera.py' shouldbe run with 'benchmark_energy = False' at line 18

# Energy measurements

This scripts assume you have an external tool to measure energy consumption.

To measure energy with picamera, the file 'benchmark_networks_picamera.py' should be run with 'benchmark_energy = True' at line 18.

To measure energy with modlib, the file 'benchmark_networks_modlib.py' should be run