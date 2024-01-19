#!/bin/bash

#/build/mopmc --help
./build/mopmc -M examples/dive_and_rise/dive_and_rise.nm -P examples/dive_and_rise/dive_and_rise_prop_100.props -Q convex -L mse -I si-gd

#./build/mopmc examples/multiple_targets/multiple_targets.pm examples/multiple_targets/multiple_targets_21c.props

#Experiment
#./build/mopmc examples/warehouse_tests/wh-5-2-2.nm examples/warehouse_tests/whouse_tasks.pctl