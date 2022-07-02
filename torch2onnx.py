import torch
from petsCNN import petsCNN
import onnx
import numpy as np


if __name__ == '__main__':
    batch_size = 1
    torch_model = petsCNN()

    check_name = 'checkpoints/t5/model_e2_acc0.902'
    checkpoint = torch.load(check_name)


    torch_model.load_state_dict(checkpoint['model_state_dict'])
    torch_model.eval()

    x = torch.randn(batch_size, 3, 300, 300, requires_grad=True)
    # x = np.random.random((batch_size, 3, 300, 300))
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,                                      # model being run
                      x,                                                # model input (or a tuple for multiple inputs)
                      'petsCNN.onnx',                                   # where to save the model (can be a file or file-like object)
                      export_params=True,                               # store the trained parameter weights inside the model file
                      opset_version=10,                                 # the ONNX version to export the model to
                      do_constant_folding=True,                         # whether to execute constant folding for optimization
                      input_names = ['input'],                          # the model's input names
                      output_names = ['output'],                        # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                    'output' : {0 : 'batch_size'}})

    onnx.checker.check_model(onnx.load('petsCNN.onnx'))

    print('Done!')
