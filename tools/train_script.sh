
export CUDA_HOME=/usr/local/cuda

# single-gpu
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml  \
    --work_dir /mnt/tmpdata/pillar/pillarid/

# multi-gpus
CUDA_VISIBLE_DEVICES=0,1 bash tools/scripts/dist_train.sh 2 \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml \
    --work_dir /mnt/tmpdata/pillar/pillarid/