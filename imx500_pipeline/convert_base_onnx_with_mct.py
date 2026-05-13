import argparse
import json
import os

import timm
import torch
import torchvision
from torchvision.io import read_image
from torchvision.models import EfficientNet_B0_Weights, MNASNet1_0_Weights, \
    MobileNet_V2_Weights, ShuffleNet_V2_X1_5_Weights, SqueezeNet1_0_Weights
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm

# print("\n---------------------------------------- USELESS TF WARNINGS FROM MCT ------------------------------------------")
import model_compression_toolkit as mct
# print("----------------------------------------------------------------------------------------------------------------\n")


# CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python3 convert_base_onnx_with_mct.py',
    description='Script used to produce onnx models with Sony\'s Model Compression Toolkit',
    epilog="""

    Example of usage: 
    python3 convert_base_onnx_with_mct.py --config "./configs/default_mct_config.json" """,

    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--config', default="./configs/default_mct_config.json", type=str,
                    help="""Config file""")

args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------

with open(args.config, 'r') as f:
    config = json.load(f)

# avoid mtc moving tensors to gpu for no reason
def false():
    return False
torch.cuda.is_available = false

# NB: MCT needs to recreate its own ONNX models


resolutions:list[list[int]] = config["target_resolutions"]

print("\nImporting networks\n\n")

networks = []

save_folder:str = config["save_folder"]
test_run:bool = config["test_run"]
repr_data_folder:str = config["repr_data_folder"] # <------ CHANGE THIS IN THE CONFIG



key = 'EfficientNetB0'
network = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval()
networks.append(
    (key, network)
)


key = 'MnasNet1.0'
network = torchvision.models.mnasnet1_0(weights=MNASNet1_0_Weights.DEFAULT).eval()
networks.append(
    (key, network)
)


key = 'MobileNetV2'
network = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
networks.append(
    (key, network)
)


key = 'RegNetX002'
network = timm.create_model('regnetx_002.pycls_in1k', pretrained=True)
networks.append(
    (key, network)
)


key = 'RegNetY002'
network = timm.create_model('regnety_002.pycls_in1k', pretrained=True)
networks.append(
    (key, network)
)


key = 'RegNetY004'
network = timm.create_model('regnety_004.pycls_in1k', pretrained=True)
networks.append(
    (key, network)
)


key = 'ShuffleNetV2x1.5'
network = torchvision.models.shufflenet_v2_x1_5(weights=ShuffleNet_V2_X1_5_Weights.DEFAULT).eval()
networks.append(
    (key, network)
)


key = 'SqueezeNetV1.0'
network = torchvision.models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).eval()
networks.append(
    (key, network)
)


pass

tensors_to_yield_base = []
images = os.listdir(repr_data_folder)

limit:int = config["limit"] # more will cause the kernel to kill it due to memory leak bugs from MCT

for i, im_name in enumerate(tqdm(images, desc="Loading image tensors", total=limit)):
    img_path = os.path.normpath(os.path.join(repr_data_folder, im_name))
    image = read_image(img_path)
    tensors_to_yield_base.append(image.to(dtype=torch.uint8).cpu())
    if i > limit:
        break

# sh file to save commands
final_txt = "#!/usr/bin/bash\n"


if test_run:
    print("\n\nPerforming test run\n\n")


for shape in resolutions:

    print(f"\nWorking with Resolution: {shape}")

    for name, model in networks:
        model_folder = os.path.normpath(os.path.join(save_folder, name))
        onnx_savepath = os.path.normpath(os.path.join(model_folder, f"{name}_{shape[0]}_{shape[1]}.onnx"))
        if not test_run and os.path.exists(onnx_savepath):
            print(f'Skipping {name} model with shape {shape} because {onnx_savepath} already exists\n')
            continue

        print(f"\n\nWorking with {name}\n\n")

        model = model.eval().cpu()


        def representative_dataset_gen():
            global tensors_to_yield_base
            global shape
            for t in tensors_to_yield_base:
                yield [resize(t.to(dtype=torch.float32), list(shape)).unsqueeze(0)]


        example_input = resize(tensors_to_yield_base[0].to(dtype=torch.float32), list(shape)).unsqueeze(0)


        target_platform_capabilities = mct.get_target_platform_capabilities(fw_name='pytorch', target_platform_name='imx500')
        quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
            in_module=model,
            representative_data_gen=representative_dataset_gen,
            target_platform_capabilities=target_platform_capabilities
        )

        os.makedirs(model_folder, exist_ok=True)

        with torch.no_grad():
            mct.exporter.pytorch_export_model(quantized_model, save_model_path=onnx_savepath,
                                              repr_dataset=representative_dataset_gen)

        final_txt_net = final_txt + f"imxconv-pt -i {name}_{shape[0]}_{shape[1]}.onnx -o {name}_{shape[0]}_{shape[1]}_for_imx500 --overwrite-output\n"

        sh_save_path = os.path.normpath(os.path.join(save_folder, name, f"{name}{shape}.sh"))
        with open(sh_save_path, 'w') as f:
            f.writelines(final_txt_net)

        pass
        if test_run:
            print("\n\nTest run finished\n\n")
            exit(0)






pass