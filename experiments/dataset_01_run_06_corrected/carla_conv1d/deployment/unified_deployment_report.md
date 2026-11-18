# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 0.513 ± 0.019 ms | perf_counter | 0.513 ± 0.019 ms | 0.293 MB | 0.000 MB | 2430.340 MB | 2.913 MB |
| **Keras FP32 (CPU)** | 23.308 ± 0.430 ms | perf_counter | 23.308 ± 0.430 ms | 1.711 MB | 0.000 MB | 417.707 MB | 0.000 MB |
| **TFLite FP16** | 0.492 ± 0.045 ms | perf_counter | 0.492 ± 0.045 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |
| **TFLite INT8** | 0.355 ± 0.008 ms | perf_counter | 0.355 ± 0.008 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |
| **ONNX FP32 (CPU)** | 0.544 ± 0.033 ms | perf_counter | 0.544 ± 0.033 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |
| **ONNX FP16 (CPU)** | 2.289 ± 0.030 ms | perf_counter | 2.289 ± 0.030 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |
| **ONNX INT8 (CPU)** | 0.366 ± 0.041 ms | perf_counter | 0.366 ± 0.041 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |
| **TensorRT FP16 (GPU)** | 0.658 ± 0.064 ms | cuda_event | 1.656 ± 0.110 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |
| **TensorRT FP32 (GPU)** | 0.634 ± 0.017 ms | cuda_event | 1.617 ± 0.030 ms | 0.000 MB | 0.000 MB | 2430.277 MB | 2.913 MB |

## 2. Classification Metrics Evaluation

| Model Format | Graph Size (MB) | Detector Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 1.198 MB | 5.923 MB | 0.5279 ± 0.0326 ms | perf_counter | 0.5279 ± 0.0326 ms | 0.8809 | 0.9807 | 0.8142 | 0.8897 | 0.9157 | 0.00% |
| **Keras FP32 (CPU)** | 1.207 MB | 5.932 MB | 23.7974 ± 2.0100 ms | perf_counter | 23.7974 ± 2.0100 ms | 0.8809 | 0.9807 | 0.8142 | 0.8897 | 0.9157 | 0.00% |
| **TFLite FP16** | 0.605 MB | 5.330 MB | 0.5001 ± 0.0205 ms | perf_counter | 0.5001 ± 0.0205 ms | 0.8817 | 0.9807 | 0.8155 | 0.8905 | 0.9150 | 0.08% |
| **TFLite INT8** | 0.312 MB | 5.036 MB | 0.3548 ± 0.0089 ms | perf_counter | 0.3548 ± 0.0089 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.8281 | 51.03% |
| **ONNX FP32 (CPU)** | 1.202 MB | 5.927 MB | 0.5616 ± 0.0344 ms | perf_counter | 0.5616 ± 0.0344 ms | 0.8809 | 0.9807 | 0.8142 | 0.8897 | 0.9157 | 0.00% |
| **ONNX FP16 (CPU)** | 0.607 MB | 5.331 MB | 2.3037 ± 0.0403 ms | perf_counter | 2.3037 ± 0.0403 ms | 0.8809 | 0.9807 | 0.8142 | 0.8897 | 0.9152 | 0.00% |
| **ONNX INT8 (CPU)** | 0.346 MB | 5.070 MB | 0.3527 ± 0.0142 ms | perf_counter | 0.3527 ± 0.0142 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.7706 | 51.03% |
| **TensorRT FP16 (GPU)** | 0.668 MB | 5.393 MB | 0.6345 ± 0.0196 ms | cuda_event | 1.6168 ± 0.0312 ms | 0.8770 | 0.9713 | 0.8155 | 0.8866 | 0.9140 | 0.71% |
| **TensorRT FP32 (GPU)** | 1.268 MB | 5.993 MB | 0.6389 ± 0.0237 ms | cuda_event | 1.6298 ± 0.0451 ms | 0.8809 | 0.9807 | 0.8142 | 0.8897 | 0.9157 | 0.00% |
