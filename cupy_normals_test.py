import cupy as cp
import cv2
import time
import datetime
from data.load_ocid import load_ocid
import numpy as np


def main():

    # Define the CuPy CUDA kernel
    normal_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void compute_normals(const float* depthImg, float* normalImg, int depthWidth, int depthHeight, int radius) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
        if ((idx < (depthWidth - radius)) && (idx >= radius) &&
            (idy < (depthHeight - radius)) && (idy >= radius)) {
    
            int a_idx = (idy - radius) * depthWidth + (idx + radius);
            int b_idx = (idy + radius) * depthWidth + (idx + radius);
            int c_idx = idy * depthWidth + (idx - radius);
    
            float depth_a = depthImg[a_idx];
            float depth_b = depthImg[b_idx];
            float depth_c = depthImg[c_idx];
    
            float a1 = -(float)radius;
            float a3 = depth_a - depth_b;
    
            float b1 = (float)-radius;
            float b2 = (float)(-radius - radius);
            float b3 = depth_c - depth_a;
    
            float v1 = -(a3 * b2);
            float v2 = (a3 * b1) - (a1 * b3);
            float v3 = (a1 * b2);
    
            float norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    
            int normal_flat_pos = idy * depthWidth * 3 + (idx * 3);
            normalImg[normal_flat_pos + 0] = v1 / norm;
            normalImg[normal_flat_pos + 1] = v2 / norm;
            normalImg[normal_flat_pos + 2] = v3 / norm;
        }
    }
    ''', 'compute_normals')


    resize_factor = 1
    radius = 1

    ocid = load_ocid(ocid_folder_path="data/OCID-dataset", limit_number_samples="all")

    # Parameters
    depth_width = 640*resize_factor
    depth_height = 480*resize_factor
    print(depth_width, depth_height)




    for d in ocid:
        depth = d["depth"]
        # depth = cv2.resize(depth, (depth_width, depth_height))
        # print(np.min(depth), np.max(depth))
        # print(depth.dtype)

        depth = cp.array(depth.astype(cp.float32)) / 1000


        # Create an output array for the normal image
        normal_img = cp.zeros((depth_height, depth_width, 3), dtype=cp.float32)

        # Calculate grid and block dimensions
        block_dim = (16, 16)
        grid_dim = ((depth_width + block_dim[0] - 1) // block_dim[0],
                    (depth_height + block_dim[1] - 1) // block_dim[1])



        time_start = datetime.datetime.now()

        # Launch the kernel
        normal_kernel(grid_dim, block_dim, (depth, normal_img, depth_width, depth_height, radius))

        time_end = datetime.datetime.now()
        time_delta = time_end - time_start
        print("Duration:", time_delta.microseconds * 0.001, "milliseconds")

        display_img = ((normal_img + 1 / 2) * 255).get().astype(np.uint8)
        cv2.imshow("normal img", display_img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
