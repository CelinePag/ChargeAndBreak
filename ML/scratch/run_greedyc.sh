#!/bin/sh
# Corrected greedy (variant CORR) on every instance set the paper's greedy used.
# Same settings as the stored greedy runs: --prune_quantile 0.95 --safety 0.1.
cd /c/Users/celinep/Documents/GitHub/ChargeAndBreak
COMMON="greedy --greedy_corrected --prune_quantile 0.95 --safety 0.1 --jobs 6 --quiet --skip-existing"
echo "== base $(date)";   python -m src.simulation.runner_dispatch "instances/*.json" $COMMON > ML/scratch/greedyc_base.log 2>&1
echo "== sens $(date)";   python -m src.simulation.runner_dispatch "instances_sens/cs_spacing_30/*.json,instances_sens/cs_spacing_100/*.json,instances_sens/charger_power_150/*.json,instances_sens/charger_power_700/*.json,instances_sens/charger_power_1000/*.json,instances_sens/battery_300/*.json,instances_sens/battery_700/*.json,instances_sens/battery_900/*.json" $COMMON > ML/scratch/greedyc_sens.log 2>&1
echo "== diesel $(date)"; python -m src.simulation.runner_dispatch "instances_sens/diesel/*.json" $COMMON --diesel > ML/scratch/greedyc_diesel.log 2>&1
echo "== grid $(date)";   python -m src.simulation.runner_dispatch "instances_sens/grid_*/*.json" $COMMON > ML/scratch/greedyc_grid.log 2>&1
echo "== done $(date)"
