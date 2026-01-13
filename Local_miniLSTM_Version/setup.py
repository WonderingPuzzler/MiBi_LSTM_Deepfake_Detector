from DeepFake_Detector_Imports import *

# For setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='prescan_2048_cuda',
    python_requires='>=3.12.0',
    ext_modules=[
        CUDAExtension(
            name='prescan_2048_cuda',
            sources=['prescan_2048_kernel.cpp', 'parallel_scan_educational.cu'],
            extra_compile_args={
                'cxx': [
                    '-std:c++17',         # C++ standard
                ],
                'nvcc': [
                    '-O3',                # Optimization level (fastest possible)
                    '--use_fast_math',    # Speed up math as much as possible
                    '-lineinfo',          # Keep for profiling
                    '--ptxas-options=-v', # Verbose PTX assembly output
                    '-arch=sm_86',        # Ampere architecture (RTX 3070)
                    '-Xcompiler', '/std:c++17',
                    '-Xcompiler', '/Zc:preprocessor',  # Use standard-conforming preprocessor

                ]
            }   
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)