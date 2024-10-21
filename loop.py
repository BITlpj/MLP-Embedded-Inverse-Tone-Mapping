from utils.META_dataset import Dataset_test
from utils.utils import online_sample_light
from utils.META_model import INF
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
import numpy as np
import time
import copy
import cv2


def loop_online_train_conv(epoch, rate, save_path, model_path, dataset_root):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set the default data type for PyTorch to float32 (single precision)
    torch.set_default_dtype(torch.float32)

    # Initialize the dataset using the testing dataset class
    # The dataset is loaded from 'dataset_root' with a specific 'rate'
    # The test list of files is provided in a 'test_list.json' file
    dataset = Dataset_test(dataset_root, rate, os.path.join(dataset_root, 'test_list.json'))

    # Create a DataLoader for the dataset with batch size 1, no shuffling, and no parallel workers
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize the model (a transformer-like model with 3 layers, 64 hidden dimensions, and 1 additional layer)
    model = INF(num_layers=3, hidden_dim=64, add_layer=1)

    # Load the pre-trained model weights from the specified 'model_path'
    # The model is loaded onto the CUDA device (GPU) with ID 0
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.cuda()  # Move the model to the GPU

    # Define the loss function as Mean Squared Error (MSE)
    loss_fn = torch.nn.MSELoss()

    # Initialize training time and reconstruction time for performance tracking
    train_time = 0
    re_time = 0

    # Iterate over the DataLoader (which provides batches of data)
    for hdr, sdr, rgb_xy, out, input, name in tqdm(dataloader):
        # Clear the GPU cache to free up memory before the next batch
        torch.cuda.empty_cache()

        # Create a deep copy of the model for online training
        online_model = copy.deepcopy(model)
        online_model = online_model.cuda()  # Move the copied model to the GPU

        # Initialize the optimizer (Adam) for gradient-based updates with a learning rate of 0.0001
        optim = torch.optim.Adam(online_model.parameters(), lr=0.0001)

        # Move data to the GPU
        hdr = hdr.cuda()
        online_model.train()  # Set the model to training mode
        input = input.cuda()
        rgb_xy = torch.tensor(rgb_xy, dtype=torch.float32)
        out = torch.tensor(out, dtype=torch.float32)
        input = torch.tensor(input, dtype=torch.float32)
        rgb_xy = rgb_xy.cuda()
        out = out.cuda()

        # Start measuring training time for this batch
        begin_train_time = time.time()

        # Resampling rate is determined by the current epoch number, with a max of 51
        resample = min(epoch / 4, 51)

        # Perform gradient-based optimization (backpropagation) for the number of specified epochs
        for _ in range(0, epoch):
            optim.zero_grad()  # Zero out the gradients before each update

            # Forward pass: get model predictions
            t_o = online_model(rgb_xy)

            # Compute the loss between predicted output and ground truth 'out'
            t_loss = loss_fn(t_o, out)

            # Backward pass: compute gradients and update model parameters
            t_loss.backward()
            optim.step()

            # Resample data during training to focus on difficult regions
            if _ % (resample + 1) == resample:
                # Resample 'rgb_xy' and 'out' based on the difference between the model's prediction and the HDR ground truth
                rgb_xy, out = online_sample_light(torch.abs(online_model(input) - hdr), input, hdr, 0.02)
                rgb_xy = rgb_xy.cuda()
                out = out.cuda()

        # Update cumulative training time
        train_time += time.time() - begin_train_time

        # Set the model to evaluation mode for inference
        online_model.eval()

        # Measure the time taken for the reconstruction (inference)
        begin_time = time.time()
        out = online_model(input)  # Perform inference
        re_time += time.time() - begin_time

        # Save the output image to the specified save_path in PNG format
        # The image name is derived from the input file name
        name_ori = os.path.join(save_path, name[0].split('/')[-1].split('.')[0] + '.png')
        print(name_ori)

        # Clip the output to ensure values are within a valid range (0 to 1)
        final = np.clip(out[0].cpu().detach(), a_min=0, a_max=1)

        # Transpose the output tensor to match the standard image format (Height, Width, Channels)
        final = np.transpose(final, (1, 2, 0))

        # Scale the image from [0, 1] to [0, 65535] (16-bit image)
        final = final * (2 ** 16 - 1)

        # Convert the image to uint16 and save it as a PNG file, using OpenCV's BGR format
        cv2.imwrite(name_ori, cv2.cvtColor(np.array(final, dtype=np.uint16), cv2.COLOR_RGB2BGR))

    # Print the average training and reconstruction time per sample
    print('train: ', train_time / dataset.__len__())
    print('reconstruct: ', re_time / dataset.__len__())

    return