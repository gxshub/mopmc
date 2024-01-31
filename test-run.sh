#!/bin/bash

#/build/mopmc --help
#./build/mopmc -m examples/dive_and_rise/dive_and_rise.nm -p examples/dive_and_rise/dive_and_rise_prop_100.props -q convex -l mse
./build/mopmc -m examples/multiobj_scheduler05.nm -p examples/multiobj_scheduler05.pctl -q achievability
