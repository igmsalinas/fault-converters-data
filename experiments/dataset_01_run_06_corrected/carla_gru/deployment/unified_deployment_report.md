# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 226.886 ± 0.209 ms | perf_counter | 226.886 ± 0.209 ms | 3.758 MB | 0.000 MB | 2102.188 MB | 12.388 MB |
| **Keras FP32 (CPU)** | 1254.754 ± 5.411 ms | perf_counter | 1254.754 ± 5.411 ms | 2.703 MB | 0.000 MB | 446.633 MB | 0.000 MB |
| **TFLite FP16** | 222.879 ± 0.331 ms | perf_counter | 222.879 ± 0.331 ms | 0.000 MB | 0.000 MB | 2098.449 MB | 12.388 MB |
| **TFLite INT8** | 161.382 ± 2.094 ms | perf_counter | 161.382 ± 2.094 ms | 0.000 MB | 0.000 MB | 2098.449 MB | 12.388 MB |
| **ONNX FP32 (CPU)** | 161.275 ± 3.290 ms | perf_counter | 161.275 ± 3.290 ms | 0.000 MB | 0.000 MB | 2098.449 MB | 12.388 MB |
| **ONNX FP16 (CPU)** | 168.403 ± 0.908 ms | perf_counter | 168.403 ± 0.908 ms | 0.000 MB | 0.000 MB | 2098.449 MB | 12.388 MB |
| **ONNX INT8 (CPU)** | 166.711 ± 0.818 ms | perf_counter | 166.711 ± 0.818 ms | 0.000 MB | 0.000 MB | 2098.449 MB | 12.388 MB |
| **TensorRT FP16 (GPU)** | 29.196 ± 0.134 ms | cuda_event | 30.294 ± 0.163 ms | 0.000 MB | 0.000 MB | 2098.449 MB | 12.388 MB |
| **TensorRT FP32 (GPU)** | 29.310 ± 0.116 ms | cuda_event | 30.428 ± 0.136 ms | 0.000 MB | 0.000 MB | 2098.477 MB | 12.388 MB |

## 2. Classification Metrics Evaluation

| Model Format | Graph Size (MB) | Detector Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 5.784 MB | 8.147 MB | 224.5202 ± 0.3092 ms | perf_counter | 224.5202 ± 0.3092 ms | 0.8139 | 0.9444 | 0.7273 | 0.8218 | 0.8678 | 0.00% |
| **Keras FP32 (CPU)** | 5.462 MB | 7.824 MB | 1254.6705 ± 6.6928 ms | perf_counter | 1254.6705 ± 6.6928 ms | 0.8139 | 0.9444 | 0.7273 | 0.8218 | 0.8678 | 0.00% |
| **TFLite FP16** | 2.911 MB | 5.273 MB | 216.8579 ± 0.2804 ms | perf_counter | 216.8579 ± 0.2804 ms | 0.8139 | 0.9429 | 0.7286 | 0.8220 | 0.8679 | 0.16% |
| **TFLite INT8** | 1.696 MB | 4.058 MB | 152.6174 ± 1.9497 ms | perf_counter | 152.6174 ± 1.9497 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.6940 | 54.57% |
| **ONNX FP32 (CPU)** | 5.457 MB | 7.819 MB | 162.2847 ± 3.3777 ms | perf_counter | 162.2847 ± 3.3777 ms | 0.8139 | 0.9444 | 0.7273 | 0.8218 | 0.8678 | 0.00% |
| **ONNX FP16 (CPU)** | 2.731 MB | 5.093 MB | 174.9955 ± 1.0414 ms | perf_counter | 174.9955 ± 1.0414 ms | 0.8131 | 0.9428 | 0.7273 | 0.8211 | 0.8680 | 0.24% |
| **ONNX INT8 (CPU)** | 5.341 MB | 7.703 MB | 161.2205 ± 0.9514 ms | perf_counter | 161.2205 ± 0.9514 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.7515 | 54.57% |
| **TensorRT FP16 (GPU)** | 10.423 MB | 12.786 MB | 29.0741 ± 0.1037 ms | cuda_event | 30.1934 ± 0.1263 ms | 0.8139 | 0.9444 | 0.7273 | 0.8218 | 0.8678 | 0.00% |
| **TensorRT FP32 (GPU)** | 10.423 MB | 12.786 MB | 29.3315 ± 0.1232 ms | cuda_event | 30.4492 ± 0.1396 ms | 0.8139 | 0.9444 | 0.7273 | 0.8218 | 0.8678 | 0.00% |
