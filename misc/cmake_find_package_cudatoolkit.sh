#!/bin/bash
cmake --find-package -DNAME=CUDAToolkit -DCOMPILER_ID=GNU -DLANGUAGE=CUDA -DMODE=COMPILE -DCMAKE_FIND_DEBUG_MODE=ON