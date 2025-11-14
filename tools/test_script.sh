
export CUDA_HOME=/usr/local/cuda

# infer
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml \
    --ckpt /mnt/tmpdata/pillar/pillarid/default/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 --infer_time

# vis
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml \
    --ckpt /mnt/tmpdata/pillar/pillarid/default/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 --vis_path ./vis_pillarid

