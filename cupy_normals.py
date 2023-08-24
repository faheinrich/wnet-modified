import cupy as cp

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

class Normal_Calculator_Cupy:
    def __init__(self, radius, height, width):
        self.radius = radius
        self.height = height
        self.width = width
        self.normal_buffer = cp.zeros((height, width, 3), dtype=cp.float32)
        # Calculate grid and block dimensions

        self.block_dim = (16, 16)
        self.grid_dim = ((width + self.block_dim[0] - 1) // self.block_dim[0],
                    (height + self.block_dim[1] - 1) // self.block_dim[1])




    def calc_normals(self, depth):
        normal_kernel(self.grid_dim, self.block_dim, (depth, self.normal_buffer, self.width, self.height, self.radius))
        return self.normal_buffer.get()
