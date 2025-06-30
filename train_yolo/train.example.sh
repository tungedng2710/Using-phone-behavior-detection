# Copy this file into `train.sh`
python train.py \
  --model-path ./weights/yolo11l.pt \
  --data-path ./datasets/mobile_phone_v1.2/data.yaml \
  --epochs 20 \
  --batch 32 \
  --device 1 \
  --pretrained \
  --auto-set-name