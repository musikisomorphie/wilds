sudo docker run -it -v /raid/jiqing/Github/wilds/:/root/wilds \
    -v /home/jiqing/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints/ \
    -v /raid/jiqing/Github/restyle-encoder/:/root/restyle \
    -v /raid/jiqing/Data/:/root/Data \
    -v /raid/jiqing/Experiment/:/root/Experiment \
    --gpus '"device=0"'  restyle