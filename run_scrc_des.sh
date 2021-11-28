#! /bin/bash
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 012 --seed 0 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_012_0 
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 012 --seed 0 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_012_0
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 012 --seed 0 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_012_0
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 012 --seed 0 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_012_0

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 012 --seed 32 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_012_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 012 --seed 32 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_012_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 012 --seed 32 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_012_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 012 --seed 32 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_012_1

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 012 --seed 512 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_012_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 012 --seed 512 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_012_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 012 --seed 512 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_012_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 012 --seed 512 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_012_2

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 012 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_012_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 012 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_012_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 012 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_012_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 012 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_012_3

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 201 --seed 0 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_201_0
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 201 --seed 0 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_201_0 
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 201 --seed 0 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_201_0
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 201 --seed 0 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_201_0

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 201 --seed 32 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_201_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 201 --seed 32 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_201_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 201 --seed 32 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_201_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 201 --seed 32 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_201_1

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 201 --seed 512 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_201_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 201 --seed 512 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_201_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 201 --seed 512 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_201_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 201 --seed 512 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_201_2

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 201 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_201_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 201 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_201_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 201 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_201_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 201 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_201_3

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 120 --seed 0 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_120_0
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 120 --seed 0 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_120_0 
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 120 --seed 0 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_120_0
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 120 --seed 0 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_120_0

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 120 --seed 32 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_120_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 120 --seed 32 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_120_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 120 --seed 32 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_120_1
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 120 --seed 32 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_120_1

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 120 --seed 512 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_120_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 120 --seed 512 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_120_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 120 --seed 512 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_120_2
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 120 --seed 512 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_120_2

python examples/run_expt.py --dataset scrc --model densenet121  --algorithm ERM       --root_dir /root/Data/ --split_scheme 120 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/ERM_scrc_120_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm deepCORAL --root_dir /root/Data/ --split_scheme 120 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/deepCORAL_scrc_120_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm groupDRO  --root_dir /root/Data/ --split_scheme 120 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/groupDRO_scrc_120_3
python examples/run_expt.py --dataset scrc --model densenet121  --algorithm IRM       --root_dir /root/Data/ --split_scheme 120 --seed 1024 --log_dir /root/Experiment/wilds/densenet121/IRM_scrc_120_3