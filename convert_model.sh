#!/bin/bash
python3 convert_onnx.py weights/superpoint_v1.pth 100 100
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_shape [1,1,100,100] --input_model output/superpoint_100x100.onnx --output_dir output --data_type FP16
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -m output/superpoint_100x100.xml -o output/superpoint_100x100.blob -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4