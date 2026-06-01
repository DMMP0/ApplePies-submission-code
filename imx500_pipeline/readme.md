# IMX500 pipeline

## Python version

This code should be executed in a Python 3.11 environment, as more up to date versions might not work with the IMX500 converter


## Conversion

The IMX500 requires a workflow a bit different from the Hailo 8L.

We use the same networks, but we first need to convert them wit the MCT quantizer and have a custom ONNX file.
At the time of writing, this step is mandatory, as base ONNX quantization is not compatible with the sensor.

## Configuration option

- target_resolutions: target resolutions for the resulting onnx model
- save_folder: save folder for the quantized ONNX models. Default: "./qmodels_for_imx500"
- repr_data_folder: representative data folder. This folder should contain only images
- test_run: perform a test run, converting only the first network at the first resolution. Default: true,
- limit: limit the sample images. Default: 1024

NB: we set a default limit of 1024 to make it comparable with the Hailo quantizer. 
However, this amount can be memory taxing, and should be lowered on devices with < 64 GB of RAM.

## Example output

This is an example of output from a test run:

``` 
Importing networks


Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading image tensors: 1025it [00:42, 24.24it/s]
representative_data_gen generates a batch size of 1 which can be slow for optimization: consider increasing the batch size


Performing test run



Working with Resolution: [112, 112]
Skipping EfficientNetB0 model with shape [112, 112] because qmodels_for_imx500/EfficientNetB0/EfficientNetB0_112_112.onnx already exists



Working with MnasNet1.0


WARNING:Model Compression Toolkit:DepthwiseConv2D is not in model.
Statistics Collection: 1026it [00:37, 27.44it/s]
Calculating quantization parameters: 100%|██████████| 101/101 [00:10<00:00,  9.82it/s]
WARNING:Model Compression Toolkit:Attribute 'metadata' not found in the model or its submodules.
/home/dmmp/PycharmProjects/ApplePies-submission-code/imx500 pipeline/.venv_imx500/lib/python3.11/site-packages/mct_quantizers/pytorch/quantizers/weights_inferable_quantizers/weights_symmetric_inferable_quantizer.py:52: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
  threshold = torch.tensor(threshold, dtype=torch.float32).to(get_working_device())


Test run finished



Process finished with exit code 0
```

## From onnx to packerOut.zip

Together with the quantized model, the previous script will produce a shell command (saved as .sh) as well.
This command will use the imx500-converter python module to produce a memory report and the packerOut.zip file to put inside the RPI

An example of output is:

``` 
2026-05-13 16:45:23,102 INFO : Running version 3.17.1 [/home/dmmp/PycharmProjects/ApplePies-submission-code/imx500_pipeline/.venv_imx500/lib/python3.11/site-packages/uni/common/logger.py:179]
2026-05-13 16:45:23,102 INFO : Converting EfficientNetB0_112_112.onnx [/home/dmmp/PycharmProjects/ApplePies-submission-code/imx500_pipeline/.venv_imx500/lib/python3.11/site-packages/uni/common/logger.py:179]
2026-05-13 16:45:27,881 INFO : Wrote outputs to /tmp/tmpu7q5cgq7/EfficientNetB0_112_112.uni-pytorch.um.pb [/home/dmmp/PycharmProjects/ApplePies-submission-code/imx500_pipeline/.venv_imx500/lib/python3.11/site-packages/uni/common/logger.py:179]
2026-05-13 16:45:27,881 INFO : Converted successfully [/home/dmmp/PycharmProjects/ApplePies-submission-code/imx500_pipeline/.venv_imx500/lib/python3.11/site-packages/uni/common/logger.py:179]
2026-05-13 16:45:28.296 INFO : CODE: [START] Starting SDSPconv
2026-05-13 16:45:40.134 INFO : ConvFe conversion finished successfully
2026-05-13 16:45:41.055 INFO : CBE component - DspConvParser has started conversion.
2026-05-13 16:45:41.145 INFO : Dsp-Dnn-Parser finished successfully !
2026-05-13 16:45:42.813 INFO : LogicModel generated successfully ! 
2026-05-13 16:45:42.814 INFO : DspConvParser runs: 1770 [msec]
2026-05-13 16:45:42.814 INFO : CBE component - DspConvParser finished conversion.
2026-05-13 16:45:56.696 INFO : Converter Backend successfully completed!
2026-05-13 16:45:56.697 INFO : Conversion time is 13.78 Seconds
2026-05-13 16:45:57.148 INFO : packer zip successfully generated under /home/dmmp/PycharmProjects/ApplePies-submission-code/imx500_pipeline/qmodels_for_imx500/EfficientNetB0/EfficientNetB0_112_112_for_imx500/packerOut.zip
2026-05-13 16:45:57.184 INFO : CODE: [OUTPUT] Output is in /home/dmmp/PycharmProjects/ApplePies-submission-code/imx500_pipeline/qmodels_for_imx500/EfficientNetB0/EfficientNetB0_112_112_for_imx500
```