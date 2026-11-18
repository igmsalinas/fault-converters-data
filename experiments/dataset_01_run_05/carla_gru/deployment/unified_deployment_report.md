# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 456.598 ± 0.964 ms | perf_counter | 456.598 ± 0.964 ms | 9.633 MB | 0.000 MB | 2058.176 MB | 13.101 MB |
| **TFLite FP16** | 457.322 ± 1.282 ms | perf_counter | 457.322 ± 1.282 ms | 4.051 MB | 0.000 MB | 2052.613 MB | 13.101 MB |
| **ONNX FP32 (CPU)** | 331.367 ± 1.126 ms | perf_counter | 331.367 ± 1.126 ms | 0.000 MB | 0.000 MB | 2048.562 MB | 13.101 MB |
| **ONNX FP16 (CPU)** | 347.588 ± 1.090 ms | perf_counter | 347.588 ± 1.090 ms | 0.000 MB | 0.000 MB | 2048.562 MB | 13.101 MB |
| **ONNX INT8 (CPU)** | 327.915 ± 1.162 ms | perf_counter | 327.915 ± 1.162 ms | 0.000 MB | 0.000 MB | 2048.562 MB | 13.101 MB |
| **TensorRT FP16 (GPU)** | 57.985 ± 0.194 ms | cuda_event | 59.164 ± 0.179 ms | 0.109 MB | 0.000 MB | 2048.672 MB | 13.101 MB |
| **TensorRT FP32 (GPU)** | 57.667 ± 0.171 ms | cuda_event | 58.834 ± 0.166 ms | 0.000 MB | 0.000 MB | 2048.586 MB | 13.101 MB |

## 2. Classification Metrics Evaluation

| Model Format | Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 11.163 MB | 453.8615 ± 1.0674 ms | perf_counter | 453.8615 ± 1.0674 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
| **TFLite FP16** | 5.615 MB | 457.1244 ± 1.2615 ms | perf_counter | 457.1244 ± 1.2615 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
| **ONNX FP32 (CPU)** | 10.805 MB | 331.4264 ± 1.8304 ms | perf_counter | 331.4264 ± 1.8304 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
| **ONNX FP16 (CPU)** | 5.405 MB | 347.9899 ± 1.1285 ms | perf_counter | 347.9899 ± 1.1285 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
| **ONNX INT8 (CPU)** | 10.761 MB | 331.7158 ± 1.0543 ms | perf_counter | 331.7158 ± 1.0543 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
| **TensorRT FP16 (GPU)** | 19.794 MB | 57.8446 ± 0.2062 ms | cuda_event | 59.0207 ± 0.1854 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
| **TensorRT FP32 (GPU)** | 19.794 MB | 57.8477 ± 0.1726 ms | cuda_event | 59.0091 ± 0.1646 ms | 0.8431 | 0.8431 | 1.0000 | 0.9149 | 0.9971 | 0.00% |
