# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 0.168 ± 0.010 ms | perf_counter | 0.168 ± 0.010 ms | 0.000 MB | 0.000 MB | 2038.898 MB | 1.201 MB |
| **Keras FP32 (CPU)** | 26.206 ± 0.337 ms | perf_counter | 26.206 ± 0.337 ms | 2.391 MB | 0.000 MB | 415.652 MB | 0.000 MB |
| **TFLite FP16** | 0.172 ± 0.012 ms | perf_counter | 0.172 ± 0.012 ms | 0.000 MB | 0.000 MB | 2039.090 MB | 1.201 MB |
| **TFLite INT8** | 0.160 ± 0.006 ms | perf_counter | 0.160 ± 0.006 ms | 0.000 MB | 0.000 MB | 2039.090 MB | 1.201 MB |
| **ONNX FP32 (CPU)** | 0.174 ± 0.013 ms | perf_counter | 0.174 ± 0.013 ms | 0.000 MB | 0.000 MB | 2039.090 MB | 1.201 MB |
| **ONNX FP16 (CPU)** | 1.084 ± 0.041 ms | perf_counter | 1.084 ± 0.041 ms | 0.000 MB | 0.000 MB | 2039.090 MB | 1.201 MB |
| **ONNX INT8 (CPU)** | 0.181 ± 0.009 ms | perf_counter | 0.181 ± 0.009 ms | 0.000 MB | 0.000 MB | 2039.090 MB | 1.201 MB |
| **TensorRT FP16 (GPU)** | 0.370 ± 0.009 ms | cuda_event | 1.339 ± 0.024 ms | 0.000 MB | 0.000 MB | 2039.090 MB | 1.201 MB |
| **TensorRT FP32 (GPU)** | 0.453 ± 0.011 ms | cuda_event | 1.421 ± 0.022 ms | 0.012 MB | 0.000 MB | 2039.125 MB | 1.201 MB |

## 2. Classification Metrics Evaluation

| Model Format | Graph Size (MB) | Detector Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 0.659 MB | 5.384 MB | 0.1705 ± 0.0121 ms | perf_counter | 0.1705 ± 0.0121 ms | 0.8707 | 0.9591 | 0.8155 | 0.8815 | 0.9161 | 0.00% |
| **Keras FP32 (CPU)** | 0.676 MB | 5.400 MB | 26.0972 ± 0.3641 ms | perf_counter | 26.0972 ± 0.3641 ms | 0.8707 | 0.9591 | 0.8155 | 0.8815 | 0.9161 | 0.00% |
| **TFLite FP16** | 0.334 MB | 5.059 MB | 0.1669 ± 0.0080 ms | perf_counter | 0.1669 ± 0.0080 ms | 0.8699 | 0.9576 | 0.8155 | 0.8809 | 0.9160 | 0.08% |
| **TFLite INT8** | 0.172 MB | 4.897 MB | 0.1617 ± 0.0104 ms | perf_counter | 0.1617 ± 0.0104 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.8496 | 49.84% |
| **ONNX FP32 (CPU)** | 0.667 MB | 5.391 MB | 0.1755 ± 0.0200 ms | perf_counter | 0.1755 ± 0.0200 ms | 0.8707 | 0.9591 | 0.8155 | 0.8815 | 0.9161 | 0.00% |
| **ONNX FP16 (CPU)** | 0.339 MB | 5.064 MB | 1.0707 ± 0.0396 ms | perf_counter | 1.0707 ± 0.0396 ms | 0.8707 | 0.9591 | 0.8155 | 0.8815 | 0.9154 | 0.00% |
| **ONNX INT8 (CPU)** | 0.218 MB | 4.943 MB | 0.1810 ± 0.0104 ms | perf_counter | 0.1810 ± 0.0104 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.7391 | 49.84% |
| **TensorRT FP16 (GPU)** | 0.461 MB | 5.186 MB | 0.3790 ± 0.0156 ms | cuda_event | 1.3611 ± 0.0377 ms | 0.8699 | 0.9576 | 0.8155 | 0.8809 | 0.9160 | 0.08% |
| **TensorRT FP32 (GPU)** | 0.725 MB | 5.450 MB | 0.4531 ± 0.0136 ms | cuda_event | 1.4204 ± 0.0247 ms | 0.8707 | 0.9591 | 0.8155 | 0.8815 | 0.9161 | 0.00% |
