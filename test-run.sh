#!/bin/bash
echo "MOPMC TEST RUN"
export checkgpu=$(which nvidia-smi)
if [ -z "$checkgpu" ]
then
 echo "GPU NOT FOUND" ; \
 ./build/mopmc -m examples/multiobj_scheduler05.nm -p examples/multiobj_scheduler05.pctl -q achievability -v standard
else
 echo "GPU USED" ; \
 ./build/mopmc -m examples/multiobj_scheduler05.nm -p examples/multiobj_scheduler05.pctl -q achievability
fi
