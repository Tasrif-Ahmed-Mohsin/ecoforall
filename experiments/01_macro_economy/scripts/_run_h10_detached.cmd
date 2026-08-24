@echo off
cd /d E:\research\project
python -u -W ignore scripts\run_phase8_horizons_v2.py --horizons 10 --n-trials 30 > data\features\horizon_10y_v2_stdout.log 2> data\features\horizon_10y_v2_stderr.log