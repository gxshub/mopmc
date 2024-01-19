#!/bin/bash

#/build/mopmc --help
./build/mopmc -M examples/dive_and_rise/dive_and_rise.nm -P examples/dive_and_rise/dive_and_rise_prop_100.props -Q convex -L mse -I si-gd

#./build/mopmc -M examples/multiobj_scheduler05.nm -P "multi(R{\"time\"}<= 13.5 [ F \"tasks_complete\" ], R{\"energy\"}<= 1.2 [  F \"tasks_complete\" ])" -Q achievability

#./build/mopmc examples/warehouse_tests/wh-5-2-2.nm examples/warehouse_tests/whouse_tasks.pctl