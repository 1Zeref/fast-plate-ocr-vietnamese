#!/bin/bash
# Tối ưu hóa TensorRT FP16 cho Fast-Plate-OCR-Vietnamese
set -e

echo '>>> [1/2] Biên dịch TensorRT Engine cho Detector (FP16)...'
trtexec --onnx=weights/plate_detector.onnx --saveEngine=weights/tensorrt\plate_detector_fp16.engine --fp16 --minShapes=images:1x3x384x384 --optShapes=images:1x3x384x384 --maxShapes=images:8x3x384x384 --memPoolSize=workspace:1024MiB

echo '>>> [2/2] Biên dịch TensorRT Engine cho CCT OCR (FP16)...'
trtexec --onnx=weights/fast_plate_ocr_vn.onnx --saveEngine=weights/tensorrt\fast_plate_ocr_vn_fp16.engine --fp16 --minShapes=plate_strips:1x3x64x256 --optShapes=plate_strips:4x3x64x256 --maxShapes=plate_strips:16x3x64x256 --memPoolSize=workspace:512MiB

echo '>>> Hoàn tất tối ưu hóa TensorRT!'
