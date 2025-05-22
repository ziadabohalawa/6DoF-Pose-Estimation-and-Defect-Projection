# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup
import os
import sys

IS_BUILD = not any(arg in sys.argv for arg in ['--name', '--version', '--description', '--author'])

if IS_BUILD:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        print(f"Found PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"PyTorch import failed: {e}")
        print("This should not happen if PyTorch is properly installed in your environment")
        sys.exit(1)
else:
    # Dummy classes for metadata-only operations
    BuildExtension = object
    CUDAExtension = lambda name, sources, **kwargs: None

code_dir = os.path.dirname(os.path.realpath(__file__))

nvcc_flags = [
    '-Xcompiler', '-O3',
    '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '--allow-unsupported-compiler'  # ← add this line
]

c_flags = ['-O3', '-std=c++14']

setup(
    name='common',
    version='0.0.1',
    ext_modules=(
        [] if not IS_BUILD else [
            CUDAExtension('common', [
                'bindings.cpp',
                'common.cu',
            ], extra_compile_args={'gcc': c_flags, 'nvcc': nvcc_flags}),
            CUDAExtension('gridencoder', [
                f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
                f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
            ], extra_compile_args={'gcc': c_flags, 'nvcc': nvcc_flags}),
        ]
    ),
    include_dirs=[
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ],
    cmdclass=(
        {} if not IS_BUILD else {'build_ext': BuildExtension}
    ),
    python_requires='>=3.7',
    zip_safe=False,
)