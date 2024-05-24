

import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data import converter
from typing import Dict
import nncf
import openvino as ov
from functools import partial
import torch
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters

ROOT = "/home/mert/Desktop/bitirme/smileyolo"
MODEL_NAME = "visdrone-best"
DATASET_NAME = "VisDrone"

model = YOLO(f"{ROOT}/weights/{MODEL_NAME}.pt")
args = get_cfg(cfg=DEFAULT_CFG)
args.data = "VisDrone.yaml"


model.export(format='OpenVINO')


model_path = Path(f"{ROOT}/weights/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
if not model_path.exists():
    model.export(format="openvino", dynamic=True, half=False)

ov_model = ov.Core().read_model(model_path)


DATA_PATH = f"{ROOT}/datasets/{DATASET_NAME}"
CFG_PATH = f"{ROOT}/datasets/visdrone.yaml"


validator = DetectionValidator(args)
print(check_det_dataset(args.data))
validator.data = check_det_dataset(args.data)
data_loader = validator.get_dataloader(f"{ROOT}/datasets/VisDrone/VisDrone2019-MOT-val", 1)

validator.is_coco = False
validator.names = model.model.names
validator.nc = model.model.model[-1].nc
validator.nm = 32
validator.process = ops.process_mask
validator.plot_masks = []


def transform_fn(data_item: Dict):
    input_tensor = validator.preprocess(data_item)["img"].numpy()
    return input_tensor

quantization_dataset = nncf.Dataset(data_loader, transform_fn)


def validation_ac(
    compiled_model: ov.CompiledModel,
    validation_loader: torch.utils.data.DataLoader,
    validator: Validator,
    num_samples: int = None,
    log=True
) -> float:
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    num_outputs = len(compiled_model.outputs)

    counter = 0
    for batch_i, batch in enumerate(validation_loader):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [
                torch.from_numpy(results[compiled_model.output(0)]),
                torch.from_numpy(results[compiled_model.output(1)]),
            ]
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
        counter += 1
    stats = validator.get_stats()
    if num_outputs == 1:
        stats_metrics = stats["metrics/mAP50-95(B)"]
    else:
        stats_metrics = stats["metrics/mAP50-95(M)"]
    if log:
        print(f"Validate: dataset length = {counter}, metric value = {stats_metrics:.3f}")

    return stats_metrics


validation_fn = partial(validation_ac, validator=validator, log=False)

quantized_model = nncf.quantize_with_accuracy_control(
    ov_model,
    quantization_dataset,
    quantization_dataset,
    validation_fn=validation_fn,
    max_drop=0.01,
    preset=nncf.QuantizationPreset.MIXED,
    subset_size=128,
    advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
        ranking_subset_size=25
    ),
)

import ipywidgets as widgets

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

core = ov.Core()
ov_config = {}
if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
quantized_compiled_model = core.compile_model(model=quantized_model, device_name=device.value, config=ov_config)
compiled_ov_model = core.compile_model(model=ov_model, device_name=device.value, config=ov_config)

pt_result = validation_ac(compiled_ov_model, data_loader, validator)
quantized_result = validation_ac(quantized_compiled_model, data_loader, validator)


print(f'[Original OpenVINO]: {pt_result:.4f}')
print(f'[Quantized OpenVINO]: {quantized_result:.4f}')


from pathlib import Path
# Set model directory
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

ir_model_path = MODEL_DIR / 'ir_model.xml'
quantized_model_path = MODEL_DIR / 'quantized_model.xml'

# Save models to use them in the commandline banchmark app
ov.save_model(ov_model, ir_model_path, compress_to_fp16=False)
ov.save_model(quantized_model, quantized_model_path, compress_to_fp16=False)


# ! benchmark_app -m $ir_model_path -shape "[1,3,640,640]" -d $device.value -api async
# ! benchmark_app -m $quantized_model_path -shape "[1,3,640,640]" -d $device.value -api async
import os
os.system(f'benchmark_app -m {ir_model_path} -shape "[1,3,640,640]" -d {device.value} -api async')
os.system(f'benchmark_app -m {quantized_model_path} -shape "[1,3,640,640]" -d {device.value} -api async')




