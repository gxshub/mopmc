#!/bin/bash

export ARCH=$(dpkg --print-architecture)
if [[ $ARCH != "amd64" ]] ; then
  tar -xvzf ./misc/lp_solve_5.5.2.11_source.tar.gz ; \
  cd ./lp_solve_5.5/lpsolve55 ; sh ccc ; cd .. ; cd .. ; \
  mv ./lp_solve_5.5/lpsolve55/bin/ux64/liblpsolve55.a ./lpSolve ; \
  mv ./lp_solve_5.5/lpsolve55/bin/ux64/liblpsolve55.so ./lpSolve
fi

cd ./build ; $CMAKE_HOME/bin/cmake .. # -Wno-dev -DCMAKE_EXPORT_COMPILE_COMMANDS=1