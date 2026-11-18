# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ONNX FP32 (CPU)** | 26.387 ± 0.477 ms | perf_counter | 26.387 ± 0.477 ms | 0.000 MB | 0.000 MB | 2030.336 MB | 4.515 MB |
| **ONNX FP16 (CPU)** | 67.740 ± 0.961 ms | perf_counter | 67.740 ± 0.961 ms | 0.012 MB | 0.000 MB | 2030.348 MB | 4.515 MB |
| **TensorRT FP16 (GPU)** | 36.489 ± 0.181 ms | cuda_event | 37.635 ± 0.218 ms | 0.008 MB | 0.000 MB | 2030.355 MB | 4.515 MB |
| **TensorRT FP32 (GPU)** | 36.558 ± 0.172 ms | cuda_event | 37.670 ± 0.191 ms | 0.000 MB | 0.000 MB | 2030.402 MB | 4.515 MB |

## 2. Classification Metrics Evaluation

| Model Format | Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ONNX FP32 (CPU)** | 3.850 MB | 25.9524 ± 0.6372 ms | perf_counter | 25.9524 ± 0.6372 ms | 0.2745 | 1.0000 | 0.1395 | 0.2449 | 0.9826 | 0.00% |
| **ONNX FP16 (CPU)** | 1.928 MB | 67.4992 ± 0.8636 ms | perf_counter | 67.4992 ± 0.8636 ms | 0.2745 | 1.0000 | 0.1395 | 0.2449 | 0.9826 | 0.00% |
| **TensorRT FP16 (GPU)** | 12.938 MB | 36.4477 ± 0.0793 ms | cuda_event | 37.5959 ± 0.1758 ms | 0.2745 | 1.0000 | 0.1395 | 0.2449 | 0.9826 | 0.00% |
| **TensorRT FP32 (GPU)** | 12.938 MB | 36.5282 ± 0.0881 ms | cuda_event | 37.6637 ± 0.0799 ms | 0.2745 | 1.0000 | 0.1395 | 0.2449 | 0.9826 | 0.00% |
