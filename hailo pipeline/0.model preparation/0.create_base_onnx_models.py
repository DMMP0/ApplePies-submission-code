"""
This script generates some ONNX models for the desired networks
"""

import os
import onnx
import torch, torchvision, timm
import argparse
import json

def save_model_to_onnx(torch_model:torch.nn.Module|None, example_input:torch.Tensor,
                       save_name:str, save_folder:str='./', onnx_proto:onnx.ModelProto|None=None):
    """
    Save the PyTorch model as ONNX
    :param torch_model: PyTorch module of the model
    :param example_input: Example input tensor
    :param save_name: save name of the model
    :param save_folder: save folder
    :param onnx_proto: start from an already existing ONNX model instead of a PyTorch one
    """
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.normpath(os.path.join(save_folder, save_name))

    if onnx_proto is None:
        if torch_model is None:
            raise ValueError('Pytorch and Proto models cannot both be None')
        # safety net
        torch_model:torch.nn.Module = torch_model.eval()
        onnx_net = torch.onnx.export(model=torch_model,
                                     args=(example_input,),
                                     external_data=False,
                                     dynamo=True,
                                     optimize=True,
                                     verify=True
                                     )
        # need to save and reload
        onnx_net.save(destination=save_path)
        onnx_net = onnx.load(save_path)
    else:
        onnx_net = onnx_proto
    onnx.save(proto=onnx_net,f=save_path)
    print("\n\n") # space the conversions
    pass


# CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python3 0.create_base_onnx_models.py',
    description='Script used to create the ONNX models',
    epilog="""

    Example of usage: 
    python3 0.create_base_onnx_models.py --config "./configs/default_hailo_quant_and_opt_config.json" """,

    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--config', default="./configs/default_hailo_quant_and_opt_config.json", type=str,
                    help="""Config file""")

args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------

with open(args.config, 'r') as f:
    config = json.load(f)

resolutions = config["target_resolutions"]

# some needs to be declared on special cases
save_folder = config['save_folder']


# NB: unfortunately, there is no current general implementation for all possible networks
#     timm and torchvision help, but better to add them manually

for resolution in resolutions:

    print(f"Working with resolution {resolution}")



    key = 'EfficientNetB0'
    network = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT).eval()
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1,3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)


    key = 'MnasNet1.0'
    network = torchvision.models.mnasnet1_0(weights=torchvision.models.MNASNet1_0_Weights.DEFAULT).eval()
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1, 3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)

    key = 'MobileNetV2'
    network = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).eval()
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1, 3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)

    key = 'RegNetX002'
    network = timm.create_model('regnetx_002.pycls_in1k', pretrained=True)
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1, 3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)
#
    key = 'RegNetY002'
    network = timm.create_model('regnety_002.pycls_in1k', pretrained=True)
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1, 3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)
#
    key = 'RegNetY004'
    network = timm.create_model('regnety_004.pycls_in1k', pretrained=True)
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1, 3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)
#
    key = 'ShuffleNetV2x1.5'
    network = torchvision.models.shufflenet_v2_x1_5(weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT).eval()
    save_model_to_onnx(torch_model=network,
                       example_input=torch.zeros((1, 3, *resolution)),
                       save_name=f'{key}_{resolution[0]}_{resolution[1]}.onnx',
                       save_folder=save_folder)

    pass

