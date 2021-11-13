#! /bin/bash

python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm ERM       --root_dir /root/Data/ --seed 0 --log_dir /root/Experiment/resnet50/ERM_rxrx1_0
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm deepCORAL --root_dir /root/Data/ --seed 0 --log_dir /root/Experiment/resnet50/deepCORAL_rxrx1_0
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm groupDRO  --root_dir /root/Data/ --seed 0 --log_dir /root/Experiment/resnet50/groupDRO_rxrx1_0
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm IRM       --root_dir /root/Data/ --seed 0 --log_dir /root/Experiment/resnet50/IRM_rxrx1_0

python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm ERM       --root_dir /root/Data/ --seed 32 --log_dir /root/Experiment/resnet50/ERM_rxrx1_1
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm deepCORAL --root_dir /root/Data/ --seed 32 --log_dir /root/Experiment/resnet50/deepCORAL_rxrx1_1
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm groupDRO  --root_dir /root/Data/ --seed 32 --log_dir /root/Experiment/resnet50/groupDRO_rxrx1_1
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm IRM       --root_dir /root/Data/ --seed 32 --log_dir /root/Experiment/resnet50/IRM_rxrx1_1

python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm ERM       --root_dir /root/Data/ --seed 512 --log_dir /root/Experiment/resnet50/ERM_rxrx1_2
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm deepCORAL --root_dir /root/Data/ --seed 512 --log_dir /root/Experiment/resnet50/deepCORAL_rxrx1_2
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm groupDRO  --root_dir /root/Data/ --seed 512 --log_dir /root/Experiment/resnet50/groupDRO_rxrx1_2
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm IRM       --root_dir /root/Data/ --seed 512 --log_dir /root/Experiment/resnet50/IRM_rxrx1_2

python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm ERM       --root_dir /root/Data/ --seed 1024 --log_dir /root/Experiment/resnet50/ERM_rxrx1_3
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm deepCORAL --root_dir /root/Data/ --seed 1024 --log_dir /root/Experiment/resnet50/deepCORAL_rxrx1_3
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm groupDRO  --root_dir /root/Data/ --seed 1024 --log_dir /root/Experiment/resnet50/groupDRO_rxrx1_3
python examples/run_expt.py --dataset rxrx1 --model resnet50 --algorithm IRM       --root_dir /root/Data/ --seed 1024 --log_dir /root/Experiment/resnet50/IRM_rxrx1_3


