import ctypes
import numpy as np

# 加载 OpenCL 动态库
cl = ctypes.CDLL("libOpenCL.so")  # Windows 上请用 "OpenCL.dll"

# 常量定义
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_READ_WRITE = 1 << 0
CL_SUCCESS = 0

# 定义 OpenCL 接口参数类型（只定义用到的）
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

# 创建平台、设备、上下文、命令队列
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

# 创建 host 数据
n = 1024
a_np = np.arange(n).astype(np.float32)
b_np = np.arange(n).astype(np.float32)
c_np = np.empty_like(a_np)

# 创建 OpenCL buffer
a_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_ONLY, a_np.nbytes, None, ctypes.byref(err)))
b_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_ONLY, b_np.nbytes, None, ctypes.byref(err)))
c_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_np.nbytes, None, ctypes.byref(err)))

# 写入数据
cl.clEnqueueWriteBuffer = cl.clEnqueueWriteBuffer if hasattr(cl, 'clEnqueueWriteBuffer') else None
if cl.clEnqueueWriteBuffer:
    cl.clEnqueueWriteBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                        ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    cl.clEnqueueWriteBuffer(queue, a_buf, True, 0, a_np.nbytes, a_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)
    cl.clEnqueueWriteBuffer(queue, b_buf, True, 0, b_np.nbytes, b_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)

# 编写内核程序
program_src = b"""
__kernel void vec_add(__global const float *a,
                      __global const float *b,
                      __global float *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

program_src_ptr = ctypes.c_char_p(program_src)
program_size = ctypes.c_size_t(len(program_src))
program = cl.clCreateProgramWithSource(context, 1, ctypes.byref(program_src_ptr), ctypes.byref(program_size), ctypes.byref(err))
cl.clBuildProgram(program, 1, devices, None, None, None)

# 创建 kernel 并设置参数
kernel = cl.clCreateKernel(program, b"vec_add", ctypes.byref(err))
cl.clSetKernelArg(kernel, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(a_buf))
cl.clSetKernelArg(kernel, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(b_buf))
cl.clSetKernelArg(kernel, 2, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(c_buf))

# 执行 kernel
global_size = (ctypes.c_size_t * 1)(n)
cl.clEnqueueNDRangeKernel(queue, kernel, 1, None, global_size, None, 0, None, None)
cl.clFinish(queue)

# 读取结果
cl.clEnqueueReadBuffer(queue, c_buf, True, 0, c_np.nbytes, c_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)

# 验证结果
print("结果是否正确：", np.allclose(c_np, a_np + b_np))
print("前10项结果：")
print(c_np[:10])
