# Prerequisites

## Python

Due to Hailo restrictions, this script should be executed in a Python 3.10 environment

## Wheels

This script requires two additional wheels to be installed:
- Hailo Dataflow Compiler 3.33.0
- HailoRT 4.23.0

They can be both found in the Hailo developer zone section (requires registration)

## Execution

At the time of writing, Hailo restricts the use of its software to specific versions of Ubuntu.
However, the code can easily run on different distributions.

Currently, there are two ways to make the script run easily:
1. Spoof the /etc/lsb-release file (requires root privileges). This method should be avoided
2. Execute the script on the virtual environment "hailo pipeline/.venv_hailo/lib/python3.10/site-packages/hailo_sdk_client/scripts/check_system_requirements.sh". Despite failing due to unsupported os, the code should run nonetheless.

If one of these steps is not performed, you will get 'first initialization error' 127 when running the hailo compiler conversion script


# Compiler conversion

The script *"2.hailo_compiler_conversions.py"* will translate the models from ONNX format into .har format.

## Configuration options

- hw_arch: Hardware architecture of reference. Default: 8l
- base_models_folder: Folder with the base ONNX models. These models will be overwritten after being modified to be compatible with Hailo. Deafult: ../base_networks
- har_folders_save_name: Save name for the .har save folder. Default: hailo_har_models
- attributes_to_delete: At the time of writing, the Hailo converter complains if extra attributes are present. If you are importing models processed with other frameworks that add extra attributes, they must be removed. Default:  \[ weight_precision, input_precision, output_precision]
- start_node_name: Name of the starting node (NN input). Default: x

## Example output

```
[info] No GPU chosen, Selected GPU 1
Modifying models for hailo

Adding attributes to models: 100%|██████████| 29/29 [00:01<00:00, 22.86it/s]

---------------------------------------------------------------------------------------------------------
[info] Translation started on ONNX model RegNetY004_112_112
[info] Restored ONNX model RegNetY004_112_112 (completion time: 00:00:00.05)
[info] Extracted ONNXRuntime meta-data for Hailo model (completion time: 00:00:00.19)
[info] Start nodes mapped from original model: 'x': 'RegNetY004_112_112/input_layer1'.
[info] End nodes mapped from original model: 'node_linear'.
[info] Translation completed on ONNX model RegNetY004_112_112 (completion time: 00:00:00.45)
[info] Saved HAR to: /home/maurizio/PycharmProjects/Submission code/hailo pipeline/hailo_har_models_8l/RegNetY004_112_112.har
---------------------------------------------------------------------------------------------------------

[...]

Process finished with exit code 0
```


# Quantization and optimization

The script *"3.quantize_and_optimize.py"* will compile the .har file into a binary executable model.

As this is a very time-consuming operation, we already include the .hef files we used for the measurements.
By default, the script executes a test run, saving the first model.
Depending on the reference setup and optimization level, this step can take ~1h per network.

Some combinations of networks and resolutions require specific scripting modifications. If this happens, you should add a special case in the code.

Once compiled, if you have the HailoRT cli, the model can be verified with the parse-hef subcommand.

As an example:

```
hailortcli parse-hef ./hailo_hef_models_8l/opt_level_2/EfficientNetB0_112_112.hef
```

with output

```
Architecture HEF was compiled for: HAILO8L
Network group name: EfficientNetB0_112_112, Multi Context - Number of contexts: 7
    Network name: EfficientNetB0_112_112/EfficientNetB0_112_112
        VStream infos:
            Input  EfficientNetB0_112_112/input_layer1 UINT8, NHWC(112x112x3)
            Output EfficientNetB0_112_112/fc33 UINT8, NC(1000)

```

## Models

Models are available at:

https://drive.google.com/drive/folders/10kaRo47p53bjKVBcVf6Rd1mtxBPT3LML?usp=sharing



## Configuration options

- hw_arch: Hardware architecture of reference. Default: 8l
- har_folder: .har load folder. Default: hailo_har_models
- skip_if_exists: skip the compilation of a .hef model if it already exists. NB: this option is ignored on test runs. Default: true
- opt_level: Optimization level. Default: 2
- cal_dataset_folder: Calibration datasets folder. Default: ../calibration_datasets_np
- hef_save_folder: .hef save folder. Default: "hailo_hef_models",
- batch_size: Batch size to compile the .hef model for. Default: 1
- sample_size: Sample size for quantization. This is inserted in the script and must be equal to the dataset size. Default: 1024,
- test_run: Perform a test run, exiting after the first network conversion. Default: true

## Example output

This is an example of output after a test run

```
[info] No GPU chosen, Selected GPU 1


NB: test run was set to true in the config, therefore only the first network will be quantized and optimized
    


Working on EfficientNetB0_112_112.har

[info] Loading model script commands to EfficientNetB0_112_112 from string
[info] Found model with 3 input channels, using real RGB images for calibration instead of sampling random data.
[info] Starting Model Optimization
[info] Model received quantization params from the hn
[info] MatmulDecompose skipped
[info] Starting Mixed Precision
[info] Model Optimization Algorithm Mixed Precision is done (completion time is 00:00:00.44)
[info] LayerNorm Decomposition skipped
[info] Starting Statistics Collector
[info] Using dataset with 1024 entries for calibration
Calibration: 100%|██████████| 1024/1024 [00:44<00:00, 22.90entries/s]
[info] Model Optimization Algorithm Statistics Collector is done (completion time is 00:00:45.40)
[info] Using dataset with 1024 entries for calibration
Calibration: 100%|██████████| 1024/1024 [00:44<00:00, 22.96entries/s]
[info] Starting Fix zp_comp Encoding
[info] Model Optimization Algorithm Fix zp_comp Encoding is done (completion time is 00:00:00.00)
[info] Matmul Equalization skipped
[info] Starting MatmulDecomposeFix
[info] Model Optimization Algorithm MatmulDecomposeFix is done (completion time is 00:00:00.00)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc14 by 5.0 bits (More than half)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc16 by 6.0 bits (More than half)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc18 by 5.0 bits (More than half)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc20 by 5.0 bits (More than half)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc26 by 6.0 bits (More than half)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc28 by 5.0 bits (More than half)
[warning] Reducing output bits of EfficientNetB0_112_112/ne_activation_fc32 by 6.0 bits (More than half)
[info] Finetune encoding skipped
[info] Bias Correction skipped
[info] Adaround skipped
[info] Starting Quantization-Aware Fine-Tuning
[info] Using dataset with 1024 entries for finetune
Epoch 1/4
1024/1024 ━━━━━━━━━━━━━━━━━━━━ 203s 115ms/step - _distill_loss_EfficientNetB0_112_112/fc33: 12.1660 - total_distill_loss: 12.1660
Epoch 2/4
1024/1024 ━━━━━━━━━━━━━━━━━━━━ 117s 114ms/step - _distill_loss_EfficientNetB0_112_112/fc33: 5.8604 - total_distill_loss: 5.8604
Epoch 3/4
1024/1024 ━━━━━━━━━━━━━━━━━━━━ 117s 114ms/step - _distill_loss_EfficientNetB0_112_112/fc33: 1.8075 - total_distill_loss: 1.8075
Epoch 4/4
1024/1024 ━━━━━━━━━━━━━━━━━━━━ 117s 114ms/step - _distill_loss_EfficientNetB0_112_112/fc33: 2.7658 - total_distill_loss: 2.7658
[info] Model Optimization Algorithm Quantization-Aware Fine-Tuning is done (completion time is 00:09:16.74)
[info] Starting Layer Noise Analysis
Full Quant Analysis: 100%|██████████| 16/16 [01:05<00:00,  4.11s/iterations]
[info] Model Optimization Algorithm Layer Noise Analysis is done (completion time is 00:01:07.61)
[info] Output layers signal-to-noise ratio (SNR): measures the quantization noise (higher is better)
[info] 	EfficientNetB0_112_112/output_layer1 SNR:	0.0004649 dB
[info] Model Optimization is done
[info] To achieve optimal performance, set the compiler_optimization_level to "max" by adding performance_param(compiler_optimization_level=max) to the model script. Note that this may increase compilation time.
[info] Loading network parameters
[info] Starting Hailo allocation and compilation flow
[info] Building optimization options for network layers...
[info] Successfully built optimization options - 3s 851ms
[info] Trying to compile the network in a single context
[info] Single context flow failed: Recoverable single context error
[info] Building optimization options for network layers...
[info] Successfully built optimization options - 4s 47ms
[info] Using Multi-context flow
[info] Resources optimization params: max_control_utilization=60%, max_compute_utilization=60%, max_compute_16bit_utilization=60%, max_memory_utilization (weights)=60%, max_input_aligner_utilization=60%, max_apu_utilization=60%
[info] Finding the best partition to contexts...
[info] Found valid partition to 4 contexts


[info] Iteration #0 - 4 contexts,
[info] Found valid partition to 5 contexts, Performance improved by 12,3%

[...]

[info] Iteration #195 - 7 contexts,
[info] Searching for a better partition...
[<==>....................................] 
[info] Partition to contexts finished successfully
[info] Partitioner finished after 343 iterations, Time it took: 7m 19s 824ms
[info] Applying selected partition to 7 contexts...
[error] Error has occured during the layer latency calculation
[info] Validating layers feasibility
[info] input_layer1: Pass
[info] avgpool1: Pass
[info] resize1: Pass
[info] dw1_sd0: Pass
[info] dw1_sd3: Pass

[...]

[info] avgpool17_d2: Pass
[info] avgpool17_fs: Pass
[info] avgpool17_d4: Pass
[info] fc33_d3: Pass
[info] Layers feasibility validated successfully
[info] Running resources allocation (mapping) flow, time per context: 59m 59s


[info] Context:0/0 Iteration 0: Mapping prepost...          
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  *          *          *          *          *          *          *          *          V       
 worker1                                                                                                  
 worker2                                                                                                  
 worker3                                                                                                  

  00:00
[info] Context:0/6 Iteration 0: Trying parallel splits...   
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0                                                                                                  
 worker1                                                                                                  
 worker2                                                                                                  
 worker3                                                                                                  

[...]

[info] EfficientNetB0_112_112_context_0 (EfficientNetB0_112_112_context_0):
Iterations: 4
Reverts on cluster mapping: 0
Reverts on inter-cluster connectivity: 0
Reverts on pre-mapping validation: 0
Reverts on split failed: 0

[...]

Reverts on cluster mapping: 0
Reverts on inter-cluster connectivity: 0
Reverts on pre-mapping validation: 0
Reverts on split failed: 0
[info] EfficientNetB0_112_112_context_0 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 93,8%               | 25%                 | 37,5%              |
[info] | cluster_1 | 43,8%               | 12,5%               | 26,6%              |
[info] | cluster_4 | 6,3%                | 1,6%                | 2,3%               |
[info] | cluster_5 | 100%                | 26,6%               | 40,6%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60,9%               | 16,4%               | 26,8%              |
[info] +-----------+---------------------+---------------------+--------------------+

[...]

[info] Successful Mapping (allocation time: 7m 55s)
[info] Compiling kernels of EfficientNetB0_112_112_context_0...
[info] Compiling kernels of EfficientNetB0_112_112_context_1...
[info] Compiling kernels of EfficientNetB0_112_112_context_2...
[info] Compiling kernels of EfficientNetB0_112_112_context_3...
[info] Compiling kernels of EfficientNetB0_112_112_context_4...
[info] Compiling kernels of EfficientNetB0_112_112_context_5...
[info] Compiling kernels of EfficientNetB0_112_112_context_6...
[info] Bandwidth of model inputs: 0.287109 Mbps, outputs: 0.00762939 Mbps (for a single frame)
[info] Bandwidth of DDR buffers: 0.0 Mbps (for a single frame)
[info] Bandwidth of inter context tensors: 4.49042 Mbps (for a single frame)
[info] Compiling kernels of EfficientNetB0_112_112_context_0...
[info] Compiling kernels of EfficientNetB0_112_112_context_1...
[info] Compiling kernels of EfficientNetB0_112_112_context_2...
[info] Compiling kernels of EfficientNetB0_112_112_context_3...
[info] Compiling kernels of EfficientNetB0_112_112_context_4...
[info] Compiling kernels of EfficientNetB0_112_112_context_5...
[info] Compiling kernels of EfficientNetB0_112_112_context_6...
[info] Bandwidth of model inputs: 0.287109 Mbps, outputs: 0.00762939 Mbps (for a single frame)
[info] Bandwidth of DDR buffers: 0.0 Mbps (for a single frame)
[info] Bandwidth of inter context tensors: 4.49042 Mbps (for a single frame)
[info] Building HEF...
[info] Successful Compilation (compilation time: 10s)


Model saved as hailo_hef_models_8l/opt_level_2/EfficientNetB0_112_112.hef




Test run completed

Process finished with exit code 0

```