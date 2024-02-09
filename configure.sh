#!/bin/bash
export ARCH=$(dpkg --print-architecture)
if [![ $ARCH == "amd64" ]] ; then
  tar -xvzf ./misc/lp_solve_5.5.2.11_source.tar.gz ; sh ./misc/lp_solve_5.5.2.11_source/lp_solve_5.5/lpsolve55/ccc ; mv ./misc/lp_solve_5.5.2.11_source/lp_solve_5.5/lpsolve55/liblpsolve55.* ./lpSolve/
fi
cd ./build ; cmake .. -Wno-dev -DCMAKE_EXPORT_COMPILE_COM
