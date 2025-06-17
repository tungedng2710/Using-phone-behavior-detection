python pipeline.py \
    --source test_samples/test1.mp4 \
    --person_model_weights weights/yolo12n.pt \
    --phone_model_weights "weights/yolo12l.pt" \
    --save_path test_samples/temp_output.mp4
