# Structure

This section is dedicated to create the base ONNX models and calibration datasets used for the quantization and optimization step

# Model creation

This step simply converts the modlels from PyTorch to ONNX with torch Dynamo.

# Calibration dataset

This step saves a defined amount of images as a big .npy tensor.

It's important to note that these .npy files can require a significant amount of storage, especially at higher resolutions.
