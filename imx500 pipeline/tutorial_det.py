import os

import model_compression_toolkit as mct
from torchvision.transforms.functional import resize
from torchvision.io import read_image
from tqdm import tqdm

import numpy as np
import torch

shapes = (
    # (160,120),
    # (120, 160),
    # (640, 480),
    # (480, 640),
    (256, 256),
    (512, 512),
    (640, 640),
    # (1920, 1080),
    # (1080, 1920),
    # (2028, 1520),
    # (1520, 2028),
    # (4056, 3040),
    # (3040,4056),
    # (160, 160),
    # (240, 240),
    # (500, 500),

    # (800, 800),
    # (850, 850),
    # (900, 900),
    # (1000, 1000),
    # (2000, 2000),
    # (3000, 3000)
)

# -------------------------------------- Face det --------------------------------------
from face_detection.eresfd_rewritten import EResFD_IMX500

batch_size = 1 # must be 1
n_iter = 500

base_folder = '/Data/CelebA/img_for_dali/train/'
images = os.listdir(base_folder)[:n_iter]

final_txt = "#!/usr/bin/bash\n"

tensors_to_yield_base = []
for url in tqdm(images, desc="Loading image tensors"):
    img_path = os.path.normpath(os.path.join(base_folder, url))
    image = read_image(img_path)
    tensors_to_yield_base.append(image.to(dtype=torch.uint8))


for shape in shapes:
    print(f"Executing for shape {shape}")

    model = EResFD_IMX500(shape).eval()
    model.load_original_eres_weights('./face_detection/weights')

    # test compatibility with fx
    from torch.fx import symbolic_trace
    m = symbolic_trace(model)

    # -------------------------------------- Face det --------------------------------------

    def representative_dataset_gen():
        global tensors_to_yield_base
        # dataloader_iterator = iter(loader)
        for t in tensors_to_yield_base:
            yield [resize(t.to(dtype=torch.float32), list(shape)).unsqueeze(0)]



    target_platform_capabilities = mct.get_target_platform_capabilities(fw_name='pytorch',target_platform_name='imx500')

    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_capabilities
    )

    print(quantization_info)

    onnx_savepath = f'qmodels_for_imx500/eresfd_{shape[0]}_{shape[1]}.onnx'
    with torch.no_grad():
        mct.exporter.pytorch_export_model(quantized_model, save_model_path=onnx_savepath,
                                          repr_dataset=representative_dataset_gen)
    final_txt += f"imxconv-pt -i eresfd_{shape[1]}_{shape[0]}.onnx -o eresfd_{shape[1]}_{shape[0]}_for_imx500 --overwrite-output\n"


with open("qmodels_for_imx500/eresfd.sh", 'w') as f:
    f.writelines(final_txt)


