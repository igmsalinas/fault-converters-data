# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ONNX FP32 (CPU)** | 14.892 ± 0.126 ms | perf_counter | 14.892 ± 0.126 ms | 0.242 MB | 0.000 MB | 2416.875 MB | 5.664 MB |
| **ONNX FP16 (CPU)** | 23.823 ± 0.176 ms | perf_counter | 23.823 ± 0.176 ms | 0.000 MB | 0.000 MB | 2416.875 MB | 5.664 MB |
| **TensorRT FP16 (GPU)** | 5.511 ± 0.064 ms | cuda_event | 6.575 ± 0.079 ms | 0.000 MB | 0.000 MB | 2416.875 MB | 5.664 MB |
| **TensorRT FP32 (GPU)** | 4.756 ± 0.056 ms | cuda_event | 5.785 ± 0.071 ms | 0.000 MB | 0.000 MB | 2416.875 MB | 5.664 MB |
| **TensorRT INT8 (GPU)** | 4.829 ± 0.059 ms | cuda_event | 5.862 ± 0.076 ms | 0.000 MB | 0.000 MB | 2416.875 MB | 5.664 MB |

## 2. Classification Metrics Evaluation

| Model Format | Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ONNX FP32 (CPU)** | 5.329 MB | 15.0158 ± 0.0693 ms | perf_counter | 15.0158 ± 0.0693 ms | 0.7451 | 1.0000 | 0.6977 | 0.8219 | 1.0000 | 0.00% |
| **ONNX FP16 (CPU)** | 2.714 MB | 23.8066 ± 0.1427 ms | perf_counter | 23.8066 ± 0.1427 ms | 0.7451 | 1.0000 | 0.6977 | 0.8219 | 1.0000 | 0.00% |
| **TensorRT FP16 (GPU)** | 3.499 MB | 5.5185 ± 0.0660 ms | cuda_event | 6.5783 ± 0.0804 ms | 0.7451 | 1.0000 | 0.6977 | 0.8219 | 1.0000 | 0.00% |
| **TensorRT FP32 (GPU)** | 5.947 MB | 4.7737 ± 0.0500 ms | cuda_event | 5.8071 ± 0.0659 ms | 0.7451 | 1.0000 | 0.6977 | 0.8219 | 1.0000 | 0.00% |
| **TensorRT INT8 (GPU)** | 6.411 MB | 4.8167 ± 0.0557 ms | cuda_event | 5.8521 ± 0.0705 ms | 0.7451 | 1.0000 | 0.6977 | 0.8219 | 1.0000 | 0.00% |
