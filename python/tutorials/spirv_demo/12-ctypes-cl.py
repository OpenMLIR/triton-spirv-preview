import ctypes
import numpy as np

# Load OpenCL shared library
cl = ctypes.CDLL("libOpenCL.so")  # Use "OpenCL.dll" on Windows

# Constant definitions
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_READ_WRITE = 1 << 0
CL_SUCCESS = 0

# Define argument types for used OpenCL functions
cl.clGetPlatformIDs.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
cl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
cl.clCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_int)]
cl.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)]
cl.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
cl.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
cl.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
cl.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint,
                                      ctypes.c_void_p, ctypes.c_void_p]
cl.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                   ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
cl.clFinish.argtypes = [ctypes.c_void_p]

# Create platform, device, context, and command queue
num_platforms = ctypes.c_uint()
cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
platforms = (ctypes.c_void_p * num_platforms.value)()
cl.clGetPlatformIDs(num_platforms.value, platforms, None)

num_devices = ctypes.c_uint()
cl.clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(num_devices))
devices = (ctypes.c_void_p * num_devices.value)()
cl.clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, None)

err = ctypes.c_int()
context = cl.clCreateContext(None, 1, devices, None, None, ctypes.byref(err))
queue = cl.clCreateCommandQueue(context, devices[0], 0, ctypes.byref(err))

# Create host data
n = 16231
a_np = np.random.rand(n).astype(np.float32)
b_np = np.random.rand(n).astype(np.float32)
c_np = np.empty_like(a_np)

# Create OpenCL buffers
a_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_ONLY, a_np.nbytes, None, ctypes.byref(err)))
b_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_ONLY, b_np.nbytes, None, ctypes.byref(err)))
c_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_np.nbytes, None, ctypes.byref(err)))

# Write input data to device
cl.clEnqueueWriteBuffer = cl.clEnqueueWriteBuffer if hasattr(cl, 'clEnqueueWriteBuffer') else None
if cl.clEnqueueWriteBuffer:
    cl.clEnqueueWriteBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                        ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    cl.clEnqueueWriteBuffer(queue, a_buf, True, 0, a_np.nbytes, a_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)
    cl.clEnqueueWriteBuffer(queue, b_buf, True, 0, b_np.nbytes, b_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)

# Define OpenCL kernel source
program_src = b"""
__kernel void vec_add(__global float *a,
                      __global float *b,
                      __global float *c,
                      int N) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    __local float temp1[1024];
    __local float temp2[1024];
    if (id < N) {
        temp1[lid] = a[id];
        temp2[lid] = b[id];
        barrier(CLK_LOCAL_MEM_FENCE);
        c[id] = a[id] + b[id];
    }
}
"""

program_src_ptr = ctypes.c_char_p(program_src)
program_size = ctypes.c_size_t(len(program_src))
program = cl.clCreateProgramWithSource(context, 1, ctypes.byref(program_src_ptr), ctypes.byref(program_size), ctypes.byref(err))
cl.clBuildProgram(program, 1, devices, None, None, None)

# Create kernel and set arguments
kernel = cl.clCreateKernel(program, b"vec_add", ctypes.byref(err))
cl.clSetKernelArg(kernel, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(a_buf))
cl.clSetKernelArg(kernel, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(b_buf))
cl.clSetKernelArg(kernel, 2, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(c_buf))
int_value = ctypes.c_int(n)
err = cl.clSetKernelArg(kernel, 3, ctypes.sizeof(int_value), ctypes.byref(int_value))

block_size = 1024
num_sm = (n + block_size - 1) // block_size

# Launch kernel
global_size = (ctypes.c_size_t * 3)(num_sm * block_size, 1, 1)
local_size = (ctypes.c_size_t * 3)(block_size, 1, 1)
launch_kenrel = cl.clEnqueueNDRangeKernel(queue, kernel, 3, None, global_size, local_size, 0, None, None)
if launch_kenrel != CL_SUCCESS:
    print('launch kenrel fail', launch_kenrel)
cl.clFinish(queue)

# Read results from device
cl.clEnqueueReadBuffer(queue, c_buf, True, 0, c_np.nbytes, c_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)

# Verify results
print("Is result correct:", np.allclose(c_np, a_np + b_np))
print("First 10 results:")
print(c_np[:10])
