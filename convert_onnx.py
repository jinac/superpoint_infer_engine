#!/usr/bin/env python
"""
Script to convert superpoint model from pytorch to onnx.
Adapted from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime
import torch

from torch import nn

import network
from superpoint_frontend import reduce_l2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Script to convert superpoint model from pytorch to onnx')
    parser.add_argument('pth_file', help="Pytorch weights file (.pth)")
    parser.add_argument('height', type=int, help="Height in pixels of input image")
    parser.add_argument('width', type=int, help="Width in pixels of input image")
    parser.add_argument('--output_dir', default="output", help="Output directory")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="batch size of input")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weights_path = args.pth_file
    batch_size = args.batch_size
    h = args.height
    w = args.width

    # Load model.
    pt_model = network.SuperPointNet()
    pytorch_total_params = sum(p.numel() for p in pt_model.parameters())
    print('Total number of params: ', pytorch_total_params)

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    pt_model.load_state_dict(torch.load(weights_path, map_location=map_location))
    pt_model.eval()

    # Create input to the model for onnx trace.
    x = torch.randn(batch_size, 1, h, w, requires_grad=True)
    torch_out = pt_model(x)
    onnx_filename = os.path.join(output_dir, "superpoint_{}x{}.onnx".format(h, w))

    # Export the model
    torch.onnx.export(pt_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      onnx_filename,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['semi', 'desc'], # the model's output names
                      )

    # Check onnx converion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(reduce_l2(to_numpy(torch_out[1])), reduce_l2(ort_outs[1]), rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # Generate config for movidius blob
    json_filename = os.path.join(output_dir, "superpoint_{}x{}.json".format(h, w))
    with open(json_filename, 'w') as f:
        f.write(
            json.dumps(
            {
            "tensors":
            [
                {       
                    "output_tensor_name": "semi",
                    "output_dimensions": torch_out[0].shape,
                    "output_entry_iteration_index": 0,
                    "output_properties_dimensions": [0],
                    "property_key_mapping": [],
                    "output_properties_type": "f16"
                },
                {       
                    "output_tensor_name": "desc",
                    "output_dimensions": torch_out[1].shape,
                    "output_entry_iteration_index": 0,
                    "output_properties_dimensions": [0],
                    "property_key_mapping": [],
                    "output_properties_type": "f16"
                },          
            ]
            },
            separators=(',', ': '), indent=2))

if __name__ == '__main__':
    main()