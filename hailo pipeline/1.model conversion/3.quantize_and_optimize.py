import os

from  hailo_sdk_client import ClientRunner

from os.path import normpath, join
from os import makedirs
import pickle
import argparse
import json

# CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python3 3.quantize_and_optimize.py',
    description='Script used to perform quantization and optimization of the hailo models',
    epilog="""

    Example of usage: 
    python3 3.quantize_and_optimize.py --config "./configs/default_hailo_quant_and_opt_config.json" """,

    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--config', default="./configs/default_hailo_quant_and_opt_config.json", type=str,
                    help="""Config file""")

args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------

with open(args.config, 'r') as f:
    config = json.load(f)

cal_dataset_folder = config["cal_dataset_folder"]



opt_level = config["opt_level"]
skip_if_exists = config["skip_if_exists"]
test_run = config["test_run"]
hailo_arch = config['hw_arch']
har_folder = config["har_folder"]

har_models_folder = f"./{har_folder}_{hailo_arch}"
save_folder = os.path.normpath(os.path.join(f"./{config['hef_save_folder']}_{hailo_arch}", f'opt_level_{opt_level}'))
makedirs(save_folder, exist_ok=True)

model_script = f"""
# Calibration and optimization settings
model_optimization_config(
    calibration, 
    batch_size={config["batch_size"]}, 
    calibset_size={config["sample_size"]}
)

model_optimization_flavor(optimization_level={opt_level})
"""

models = os.listdir(har_models_folder)
models.sort()

if test_run:
    print("""

NB: test run was set to true in the config, therefore only the first network will be quantized and optimized
    
""")

for model_name in models:

    model_name_no_ext = '.'.join(model_name.split('.')[:-1])
    model_name_no_res = model_name_no_ext.split('_')[:-2][0]
    res = [int(r) for r in model_name_no_ext.split('_')[-2:]]
    hef_save_name = f"{model_name_no_ext}.hef"

    save_path = normpath(join(save_folder, hef_save_name))
    if not test_run and skip_if_exists and os.path.exists(save_path):
        print(f"{hef_save_name} skipped because it already exists\n")
        continue


    # todo: remove and retry known problematic resolutions
    if res[1] == 640 or res[1] == 512:
        print(f'TODO: REMOVE THIS TEMPORARY SKIP ({model_name_no_ext})')
        continue



    pass

    # add model specific opts:
    if model_name_no_res == 'EfficientNetB0':

        if model_name_no_ext == 'EfficientNetB0_256_256':
            pass
            model_script_modded = model_script + """
# try to solve quantization problems
pre_quantization_optimization(global_avgpool_reduction, layers=[avgpool1, avgpool2, avgpool3], division_factors=[2, 2])
        """
        elif res[1] == 640 or res[1] == 512     :
            print(f'Skipping {model_name_no_ext} due to conflicting hailo error messages')
            continue

        else:

            model_script_modded = model_script + """
# try to solve quantization problems
pre_quantization_optimization(global_avgpool_reduction, layers=[avgpool1, avgpool2], division_factors=[2, 2])
"""
        pass
    elif model_name_no_res == 'RegNetY004':
        if res[1] == 640:
            model_script_modded = model_script + """
# try to solve quantization problems
pre_quantization_optimization(global_avgpool_reduction, layers=[avgpool1, avgpool2, avgpool3], division_factors=[4, 4])
"""
        elif res[1] == 512:
            print(f'Skipping {model_name_no_ext} due to conflicting hailo error messages')
            continue
        else:
            model_script_modded = model_script + """
# try to solve quantization problems
pre_quantization_optimization(global_avgpool_reduction, layers=[avgpool1], division_factors=[2, 2])
"""
    elif model_name_no_res == 'RegNetX002':
        if res[1] == 640 or res[1] == 512:
            pass
            model_script_modded = model_script + """
# try to solve quantization problems
pre_quantization_optimization(global_avgpool_reduction, layers=[avgpool1], division_factors=[4, 4])
                """
        else:
            model_script_modded = model_script

    elif model_name_no_res == 'EfficientNetV2':
        model_script_modded = model_script + """
# Let each channel have its own scale/zero-point
model_optimization_config(globals, output_encoding_vector=enabled)
allocator_param(enable_muxer=False)
"""
        # TODO: hailo_model_optimization.acceleras.utils.acceleras_exceptions.NegativeSlopeExponentNonFixable: Quantization failed in layer EfficientNetV2_224_224/ne_activation_fc50 due to unsupported required slope. Desired shift is 11.0, but op has only 8 data bits. This error raises when the data or weight range are not balanced. Mostly happens when using random calibration-set/weights, the calibration-set is not normalized properly or batch-normalization was not used during training.
        continue

    else:
        model_script_modded = model_script
    print(f"\nWorking on {model_name}\n")

    cal_data_name = f"cal_data_{res[0]}_{res[1]}.npy"
    cal_data_path = normpath(join(cal_dataset_folder, cal_data_name))

    with open(cal_data_path, 'rb') as f:
        cal_data = pickle.load(f)


    model_path = normpath(join(har_models_folder, model_name))

    runner = ClientRunner(har=model_path, hw_arch=f'hailo{hailo_arch}')
    runner.load_model_script(model_script_modded)
    runner.optimize(cal_data)

    pass

    hef_binary = runner.compile()

    pass


    with open(save_path, 'wb') as f:
        f.write(hef_binary)

    print(f"\n\nModel saved as {save_path}\n\n")
    pass
    if test_run:
        print("\n\nTest run completed")
        exit(0)