# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 0.602 ± 0.028 ms | perf_counter | 0.602 ± 0.028 ms | 0.426 MB | 0.000 MB | 1932.359 MB | 2.839 MB |
| **TFLite FP16** | 0.590 ± 0.036 ms | perf_counter | 0.590 ± 0.036 ms | 0.000 MB | 0.000 MB | 1932.180 MB | 2.839 MB |
| **TFLite INT8** | 0.341 ± 0.012 ms | perf_counter | 0.341 ± 0.012 ms | 0.000 MB | 0.000 MB | 1932.180 MB | 2.839 MB |
| **ONNX FP32 (CPU)** | 0.613 ± 0.021 ms | perf_counter | 0.613 ± 0.021 ms | 0.254 MB | 0.000 MB | 1932.434 MB | 2.839 MB |
| **ONNX FP16 (CPU)** | 3.383 ± 0.048 ms | perf_counter | 3.383 ± 0.048 ms | 0.715 MB | 0.000 MB | 1933.148 MB | 2.839 MB |
| **ONNX INT8 (CPU)** | 0.276 ± 0.016 ms | perf_counter | 0.276 ± 0.016 ms | 0.000 MB | 0.000 MB | 1933.148 MB | 2.839 MB |
| **TensorRT FP16 (GPU)** | 0.378 ± 0.011 ms | cuda_event | 1.370 ± 0.025 ms | 0.062 MB | 0.000 MB | 1933.211 MB | 2.839 MB |
| **TensorRT FP32 (GPU)** | 0.462 ± 0.012 ms | cuda_event | 1.449 ± 0.024 ms | 0.012 MB | 0.000 MB | 1933.246 MB | 2.839 MB |
| **TensorRT INT8 (GPU)** | 0.463 ± 0.011 ms | cuda_event | 1.448 ± 0.024 ms | 0.012 MB | 0.000 MB | 1933.281 MB | 2.839 MB |

## 2. Classification Metrics Evaluation

| Model Format | Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 1.925 MB | 0.5983 ± 0.0286 ms | perf_counter | 0.5983 ± 0.0286 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **TFLite FP16** | 0.966 MB | 0.5895 ± 0.0379 ms | perf_counter | 0.5895 ± 0.0379 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **TFLite INT8** | 0.490 MB | 0.3385 ± 0.0083 ms | perf_counter | 0.3385 ± 0.0083 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9942 | 0.00% |
| **ONNX FP32 (CPU)** | 1.939 MB | 0.5859 ± 0.0224 ms | perf_counter | 0.5859 ± 0.0224 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **ONNX FP16 (CPU)** | 0.975 MB | 3.3915 ± 0.0228 ms | perf_counter | 3.3915 ± 0.0228 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **ONNX INT8 (CPU)** | 0.546 MB | 0.2767 ± 0.0138 ms | perf_counter | 0.2767 ± 0.0138 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **TensorRT FP16 (GPU)** | 0.994 MB | 0.3796 ± 0.0130 ms | cuda_event | 1.3776 ± 0.0312 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **TensorRT FP32 (GPU)** | 1.991 MB | 0.4675 ± 0.0102 ms | cuda_event | 1.4603 ± 0.0250 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
| **TensorRT INT8 (GPU)** | 2.007 MB | 0.4712 ± 0.0133 ms | cuda_event | 1.4625 ± 0.0243 ms | 0.9608 | 0.9767 | 0.9767 | 0.9767 | 0.9971 | 0.00% |
