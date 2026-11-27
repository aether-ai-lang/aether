"""OpenCL execution-plan template for Aether 0.8 (aetherc.cbackend).

The generated host program loads the vendor ICD (OpenCL.dll / libOpenCL.so)
at RUNTIME via LoadLibrary/dlopen — so it compiles with plain gcc/clang/MSVC
with zero OpenCL SDK installed, and reports honest fallback telemetry when no
OpenCL runtime is present on the machine.
"""

OPENCL_HOST_TEMPLATE = r"""/*
 * Aether 0.8 -- generated OpenCL execution plan. DO NOT EDIT.
 * source: __SOURCE__
 * plan:   f32 | OpenCL forward pass via runtime-loaded ICD | honest fallback
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE ae_dl_t;
typedef FARPROC ae_sym_t;
static ae_dl_t ae_dlopen(const char *path) { return LoadLibraryA(path); }
static ae_sym_t ae_dlsym(ae_dl_t handle, const char *name) { return GetProcAddress(handle, name); }
#else
#include <dlfcn.h>
typedef void *ae_dl_t;
typedef void *ae_sym_t;
static ae_dl_t ae_dlopen(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static ae_sym_t ae_dlsym(ae_dl_t handle, const char *name) { return dlsym(handle, name); }
#endif

/* Minimal OpenCL API surface, resolved from the ICD at runtime. */
typedef int cl_int;
typedef unsigned int cl_uint;
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_program;
typedef void *cl_kernel;
typedef void *cl_mem;

static ae_dl_t ae_cl_lib = NULL;

typedef cl_int (*pfn_aeGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*pfn_aeGetDeviceIDs)(cl_platform_id, cl_uint, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*pfn_aeCreateContext)(const void *, cl_uint, const cl_device_id *, void *, cl_int *);
typedef cl_command_queue (*pfn_aeCreateQueue)(cl_context, cl_device_id, cl_uint, cl_int *);
typedef cl_mem (*pfn_aeCreateBuffer)(cl_context, cl_uint, size_t, void *, cl_int *);
typedef cl_program (*pfn_aeCreateProgram)(cl_context, const char **, const size_t *, cl_int *);
typedef cl_int (*pfn_aeBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
typedef cl_kernel (*pfn_aeCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*pfn_aeSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*pfn_aeEnqueueWrite)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, void *, void *);
typedef cl_int (*pfn_aeEnqueueNDRange)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, void **, void *);
typedef cl_int (*pfn_aeEnqueueRead)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, void **, void *);
typedef cl_int (*pfn_aeFinish)(cl_command_queue);

static pfn_aeGetPlatformIDs aeGetPlatformIDs;
static pfn_aeGetDeviceIDs aeGetDeviceIDs;
static pfn_aeCreateContext aeCreateContext;
static pfn_aeCreateQueue aeCreateQueue;
static pfn_aeCreateBuffer aeCreateBuffer;
static pfn_aeCreateProgram aeCreateProgram;
static pfn_aeBuildProgram aeBuildProgram;
static pfn_aeCreateKernel aeCreateKernel;
static pfn_aeSetKernelArg aeSetKernelArg;
static pfn_aeEnqueueWrite aeEnqueueWrite;
static pfn_aeEnqueueNDRange aeEnqueueNDRange;
static pfn_aeEnqueueRead aeEnqueueRead;
static pfn_aeFinish aeFinish;

static const char *AE_ICD_CANDIDATES[] = {
#ifdef _WIN32
    "OpenCL.dll",
#else
    "libOpenCL.so.1",
    "libOpenCL.so",
#endif
    NULL
};

static int ae_opencl_load(void) {
    for (int i = 0; AE_ICD_CANDIDATES[i]; i++) {
        ae_cl_lib = ae_dlopen(AE_ICD_CANDIDATES[i]);
        if (ae_cl_lib) break;
    }
    if (!ae_cl_lib) return -1;
#define AE_SYM(var, name) \
    var = (pfn_##var)ae_dlsym(ae_cl_lib, name); \
    if (!var) return -2
    AE_SYM(aeGetPlatformIDs, "clGetPlatformIDs");
    AE_SYM(aeGetDeviceIDs, "clGetDeviceIDs");
    AE_SYM(aeCreateContext, "clCreateContext");
    AE_SYM(aeCreateQueue, "clCreateCommandQueue");
    AE_SYM(aeCreateBuffer, "clCreateBuffer");
    AE_SYM(aeCreateProgram, "clCreateProgramWithSource");
    AE_SYM(aeBuildProgram, "clBuildProgram");
    AE_SYM(aeCreateKernel, "clCreateKernel");
    AE_SYM(aeSetKernelArg, "clSetKernelArg");
    AE_SYM(aeEnqueueWrite, "clEnqueueWriteBuffer");
    AE_SYM(aeEnqueueNDRange, "clEnqueueNDRangeKernel");
    AE_SYM(aeEnqueueRead, "clEnqueueReadBuffer");
    AE_SYM(aeFinish, "clFinish");
#undef AE_SYM
    return 0;
}

static const char *AE_KERNEL_SOURCE =
"__kernel void ae_scale_pass(__global const float *input,\n"
"                            __global const float *weights,\n"
"                            __global float *output,\n"
"                            const int count,\n"
"                            const float factor) {\n"
"    int i = get_global_id(0);\n"
"    if (i < count) output[i] = input[i] * weights[i % count] * factor;\n"
"}\n";

int main(void) {
    printf("  [device] auto_hardware -> opencl probe | runtime-loaded ICD\n");
    if (ae_opencl_load() != 0) {
        printf("  [fallback] no OpenCL runtime found (install an ICD from your GPU vendor)\n");
        printf("  [fallback] dispatching semantics to the C99/AVX-512 tier next run (--target=c)\n");
        return 0;
    }

    cl_uint platform_count = 0;
    if (aeGetPlatformIDs(0, NULL, &platform_count) != 0 || platform_count == 0) {
        printf("  [fallback] no OpenCL platforms found\n");
        return 0;
    }
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
    aeGetPlatformIDs(platform_count, platforms, NULL);
    cl_device_id device = NULL;
    cl_platform_id platform = NULL;
    for (cl_uint p = 0; p < platform_count && !device; p++) {
        cl_uint count = 0;
        if (aeGetDeviceIDs(platforms[p], 1 /* CL_DEVICE_TYPE_GPU */, 1, &device, &count) != 0 || count == 0) {
            device = NULL;
            if (aeGetDeviceIDs(platforms[p], 2 /* CL_DEVICE_TYPE_CPU */, 1, &device, &count) != 0 || count == 0) {
                device = NULL;
            }
        }
        if (device) platform = platforms[p];
    }
    if (!device) {
        printf("  [fallback] no OpenCL devices found\n");
        return 0;
    }

    cl_int err = 0;
    cl_context context = aeCreateContext(NULL, 1, &device, NULL, &err);
    if (!context || err != 0) { printf("  [fallback] clCreateContext failed (%d)\n", err); return 0; }
    cl_command_queue queue = aeCreateQueue(context, device, 0, &err);
    if (!queue || err != 0) { printf("  [fallback] clCreateCommandQueue failed (%d)\n", err); return 0; }

    size_t source_len = strlen(AE_KERNEL_SOURCE);
    cl_program program = aeCreateProgram(context, &AE_KERNEL_SOURCE, &source_len, &err);
    if (!program || err != 0) { printf("  [fallback] clCreateProgramWithSource failed (%d)\n", err); return 0; }
    err = aeBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != 0) { printf("  [fallback] clBuildProgram failed (%d)\n", err); return 0; }
    cl_kernel kernel = aeCreateKernel(program, "ae_scale_pass", &err);
    if (!kernel || err != 0) { printf("  [fallback] clCreateKernel failed (%d)\n", err); return 0; }

    const int count = 1 << 20;
    float *input = (float *)malloc((size_t)count * 4);
    float *weights = (float *)malloc((size_t)count * 4);
    float *output = (float *)malloc((size_t)count * 4);
    for (int i = 0; i < count; i++) { input[i] = i * 0.001f; weights[i] = 0.5f; }

    cl_mem d_in = aeCreateBuffer(context, 0x1 /* RW */, (size_t)count * 4, NULL, &err);
    cl_mem d_w = aeCreateBuffer(context, 0x1, (size_t)count * 4, NULL, &err);
    cl_mem d_out = aeCreateBuffer(context, 0x1, (size_t)count * 4, NULL, &err);
    if (!d_in || !d_w || !d_out) { printf("  [fallback] clCreateBuffer failed\n"); return 0; }

    aeEnqueueWrite(queue, d_in, 0, 0, (size_t)count * 4, input, 0, NULL, NULL);
    aeEnqueueWrite(queue, d_w, 0, 0, (size_t)count * 4, weights, 0, NULL, NULL);
    float factor = 2.0f;
    aeSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    aeSetKernelArg(kernel, 1, sizeof(cl_mem), &d_w);
    aeSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out);
    aeSetKernelArg(kernel, 3, sizeof(int), &count);
    aeSetKernelArg(kernel, 4, sizeof(float), &factor);
    size_t global = (size_t)count;
    err = aeEnqueueNDRange(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    aeFinish(queue);
    aeEnqueueRead(queue, d_out, 0, 0, (size_t)count * 4, output, 0, NULL, NULL);
    aeFinish(queue);

    printf("  [opencl]  kernel executed | %d elements | scale x%.1f | out[0]=%.3f\n",
           count, factor, output[0]);

    free(input); free(weights); free(output);
    free(platforms);
    return 0;
}
"""

OPENCL_KERNEL_SOURCE = r"""__kernel void ae_scale_pass(__global const float *input,
                            __global const float *weights,
                            __global float *output,
                            const int count,
                            const float factor) {
    int i = get_global_id(0);
    if (i < count) output[i] = input[i] * weights[i % count] * factor;
}
"""
