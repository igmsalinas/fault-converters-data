# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 22.128 ± 0.069 ms | perf_counter | 22.128 ± 0.069 ms | 12.305 MB | 0.000 MB | 2377.273 MB | 15.949 MB |
| **TFLite FP16** | 26.665 ± 0.131 ms | perf_counter | 26.665 ± 0.131 ms | 6.809 MB | 0.000 MB | 2372.016 MB | 15.949 MB |
| **TFLite INT8** | 10.098 ± 0.034 ms | perf_counter | 10.098 ± 0.034 ms | 0.629 MB | 0.000 MB | 2366.305 MB | 15.949 MB |
| **ONNX FP32 (CPU)** | 12.248 ± 0.073 ms | perf_counter | 12.248 ± 0.073 ms | 6.953 MB | 0.000 MB | 2372.629 MB | 15.949 MB |
| **ONNX FP16 (CPU)** | 30.800 ± 0.083 ms | perf_counter | 30.800 ± 0.083 ms | 0.512 MB | 0.000 MB | 2373.141 MB | 15.949 MB |
| **ONNX INT8 (CPU)** | 21.310 ± 0.345 ms | perf_counter | 21.310 ± 0.345 ms | 0.000 MB | 0.000 MB | 2373.254 MB | 15.949 MB |
| **TensorRT FP16 (GPU)** | 2.607 ± 0.031 ms | cuda_event | 3.674 ± 0.058 ms | 0.051 MB | 0.000 MB | 2373.305 MB | 15.949 MB |
| **TensorRT FP32 (GPU)** | 2.823 ± 0.030 ms | cuda_event | 3.884 ± 0.053 ms | 0.000 MB | 0.000 MB | 2373.305 MB | 15.949 MB |

## 2. Classification Metrics Evaluation

| Model Format | Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 12.869 MB | 22.7711 ± 0.0766 ms | perf_counter | 22.7711 ± 0.0766 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **TFLite FP16** | 6.448 MB | 28.2979 ± 0.0618 ms | perf_counter | 28.2979 ± 0.0618 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **TFLite INT8** | 3.293 MB | 9.9172 ± 0.0434 ms | perf_counter | 9.9172 ± 0.0434 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **ONNX FP32 (CPU)** | 12.880 MB | 12.4704 ± 0.0727 ms | perf_counter | 12.4704 ± 0.0727 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **ONNX FP16 (CPU)** | 6.452 MB | 30.7134 ± 0.0865 ms | perf_counter | 30.7134 ± 0.0865 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **ONNX INT8 (CPU)** | 3.394 MB | 21.2720 ± 0.3258 ms | perf_counter | 21.2720 ± 0.3258 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **TensorRT FP16 (GPU)** | 10.416 MB | 2.6065 ± 0.0272 ms | cuda_event | 3.6743 ± 0.0534 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
| **TensorRT FP32 (GPU)** | 14.642 MB | 2.8234 ± 0.0429 ms | cuda_event | 3.8852 ± 0.0658 ms | 0.8039 | 1.0000 | 0.7674 | 0.8684 | 1.0000 | 0.00% |
