#! /bin/bash

python examples/run_expt.py --dataset scrc --algorithm deepCORAL --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_deepCORAL_scrc_0
python examples/run_expt.py --dataset scrc --algorithm groupDRO --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_groupDRO_scrc_0
python examples/run_expt.py --dataset scrc --algorithm IRM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_IRM_scrc_0
python examples/run_expt.py --dataset scrc --algorithm ERM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_ERM_scrc_0

python examples/run_expt.py --dataset scrc --algorithm deepCORAL --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_deepCORAL_scrc_1
python examples/run_expt.py --dataset scrc --algorithm groupDRO --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_groupDRO_scrc_1
python examples/run_expt.py --dataset scrc --algorithm IRM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_IRM_scrc_1
python examples/run_expt.py --dataset scrc --algorithm ERM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_ERM_scrc_1

python examples/run_expt.py --dataset scrc --algorithm deepCORAL --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_deepCORAL_scrc_2
python examples/run_expt.py --dataset scrc --algorithm groupDRO --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_groupDRO_scrc_2
python examples/run_expt.py --dataset scrc --algorithm IRM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_IRM_scrc_2
python examples/run_expt.py --dataset scrc --algorithm ERM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_ERM_scrc_2

python examples/run_expt.py --dataset scrc --algorithm deepCORAL --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_deepCORAL_scrc_3
python examples/run_expt.py --dataset scrc --algorithm groupDRO --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_groupDRO_scrc_3
python examples/run_expt.py --dataset scrc --algorithm IRM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_IRM_scrc_3
python examples/run_expt.py --dataset scrc --algorithm ERM --root_dir /home/histopath/Data/SCRC_nuclei/ --log_dir ./logs_ERM_scrc_3