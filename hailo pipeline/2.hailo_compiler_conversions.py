import os

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python' # uncomment this if you have problems with protobuf

from hailo_sdk_client import ClientRunner

from os import makedirs
from os.path import normpath, join
import onnx
from tqdm import tqdm
import argparse
import json

# CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python3 2.hailo_compiler_conversions.py',
    description='Script used to create the ONNX models',
    epilog="""

    Example of usage: 
    python3 2.hailo_compiler_conversions.py --config "./configs/default_hailo_conversion_config.json" """,

    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--config', default="./configs/default_hailo_conversion_config.json", type=str,
                    help="""Config file""")

args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------

with open(args.config, 'r') as f:
    config = json.load(f)

hw_arch = config['hw_arch']

runner = ClientRunner(hw_arch=f'hailo{hw_arch}')

base_models_folder:str = config["base_models_folder"]
har_folders_save_name:str = config['har_folders_save_name']
save_folder = f"{har_folders_save_name}_{hw_arch}"
makedirs(save_folder, exist_ok=True)

models = os.listdir(base_models_folder)

attributes_to_delete:list[str] = config["attributes_to_delete"]
start_node_name = config["start_node_name"]

print("\nModifying models for hailo\n")

for m in tqdm(models, desc='Adding attributes to models'):

    model_name = m
    model_path = normpath(join(base_models_folder, model_name))

    # ensure onnx
    if model_path[-5:] != '.onnx':
        continue


    model = onnx.load(model_path)


    graph_def = model.graph
    initializers = graph_def.initializer
    pass
    nodes = graph_def.node

    to_del = set(attributes_to_delete)

    for node in nodes:
        to_add = {
            'kernel_shape': False,
        }
        # must remove some stuff
        to_rem = []
        for i, att in enumerate(node.attribute):
            if att.name in to_del:
                to_rem.append(i)
            if att.name in to_add:
                to_add[att.name] = to_add[att.name] or att.name in to_add
        to_rem.reverse()
        for i in to_rem:
            _ = node.attribute.pop(i)
        # idw = node.attribute.index(value='weight_precision')
        if node.op_type == "Conv":
            if to_add['kernel_shape']:
                continue
            # input 2 = weights
            w = node.input[1]
            # search in initializers
            wanted_i = None

            for ini in initializers:
                if ini.name == w:
                    wanted_i = ini.dims
                    break
            if wanted_i is None:
                raise NotImplementedError() # todo
            # Add kernel shape if not present
            kernel_attribute = onnx.helper.make_attribute("kernel_shape", wanted_i)
            node.attribute.extend([kernel_attribute])

    # infer shapes
    model = onnx.shape_inference.infer_shapes(model=model)
    onnx.checker.check_model(model) # check again
    onnx.save(model, model_path)

pass

for m in models:

    # check onnx
    if m[-5:] != ".onnx":
        continue

    print('\n---------------------------------------------------------------------------------------------------------')
    model_path = normpath(join(base_models_folder, m))
    model_name = m[:-5]

    res = model_name.split('_')[-2:]
    res=(int(res[0]), int(res[1]))
    try:
        hn, calibration_data = runner.translate_onnx_model(
            model=model_path,           # path to your ONNX model
            net_name=model_name,               # a name for the model (optional)
            start_node_names=[start_node_name],       # name of the input node in the ONNX graph (optional)
            #end_node_names=enn,        # name of the output node (optional)
            net_input_shapes={start_node_name: [1, 3, res[0], res[1]]},  # input shape [N,C,H,W],
            disable_shape_inference = True,
            disable_rt_metadata_extraction = False
        )
    except:
        # skip for now.
        # this should never happen, if it does, put a breakpoint
        print(f'Skipped {model_name} due to hailo error')
        continue


    runner.save_har(normpath(join(save_folder,f"{model_name}.har")))
    print('---------------------------------------------------------------------------------------------------------\n')