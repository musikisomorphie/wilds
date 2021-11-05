sudo docker run -it -v /home/histopath/Github/wilds/:/root/wilds \
    -v /home/histopath/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints/ \
    -v /home/histopath/Github/restyle-encoder/:/root/restyle \
    -v /home/histopath/Data/:/root/Data \
    -v /home/histopath/Experiment/:/root/Experiment \
    --shm-size 8G --gpus all  restyle