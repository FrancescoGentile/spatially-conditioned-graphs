##
##
##

# python main.py \
#     --world-size 1 \
#     --dataset h2o \
#     --partitions "train" "test" \
#     --data-root "datasets/h2o" \
#     --train-detection-dir "detections/h2o/train" \
#     --val-detection-dir "detections/h2o/test" \
#     --batch-size 1


python main.py \
    --world-size 1 \
    --dataset hicodet \
    --partitions "train2015" "test2015" \
    --data-root "hicodet/" \
    --train-detection-dir "hicodet/detections/train2015_gt" \
    --val-detection-dir "hicodet/detections/test2015_gt" \
    --batch-size 2
