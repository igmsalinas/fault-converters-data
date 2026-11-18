# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 46.766 ± 0.690 ms | perf_counter | 46.766 ± 0.690 ms | 0.000 MB | 0.000 MB | 2124.676 MB | 4.994 MB |
| **Keras FP32 (CPU)** | 1033.690 ± 7.766 ms | perf_counter | 1033.690 ± 7.766 ms | 2.352 MB | 0.000 MB | 473.586 MB | 0.000 MB |
| **TFLite FP16** | 45.746 ± 1.131 ms | perf_counter | 45.746 ± 1.131 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |
| **TFLite INT8** | 26.573 ± 0.235 ms | perf_counter | 26.573 ± 0.235 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |
| **ONNX FP32 (CPU)** | 14.772 ± 0.550 ms | perf_counter | 14.772 ± 0.550 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |
| **ONNX FP16 (CPU)** | 39.244 ± 0.435 ms | perf_counter | 39.244 ± 0.435 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |
| **ONNX INT8 (CPU)** | 17.744 ± 0.428 ms | perf_counter | 17.744 ± 0.428 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |
| **TensorRT FP16 (GPU)** | 19.814 ± 0.601 ms | cuda_event | 20.919 ± 0.614 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |
| **TensorRT FP32 (GPU)** | 19.904 ± 0.815 ms | cuda_event | 21.012 ± 0.827 ms | 0.000 MB | 0.000 MB | 2124.695 MB | 4.994 MB |

## 2. Classification Metrics Evaluation

| Model Format | Graph Size (MB) | Detector Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 2.238 MB | 6.963 MB | 46.9410 ± 1.3509 ms | perf_counter | 46.9410 ± 1.3509 ms | 0.8368 | 0.9486 | 0.7647 | 0.8468 | 0.8853 | 2.29% |
| **Keras FP32 (CPU)** | 2.239 MB | 6.963 MB | 1032.7817 ± 8.1229 ms | perf_counter | 1032.7817 ± 8.1229 ms | 0.8336 | 0.9468 | 0.7607 | 0.8436 | 0.8835 | 0.00% |
| **TFLite FP16** | 1.128 MB | 5.853 MB | 46.5461 ± 0.6450 ms | perf_counter | 46.5461 ± 0.6450 ms | 0.8131 | 0.8604 | 0.8155 | 0.8373 | 0.8781 | 9.54% |
| **TFLite INT8** | 0.575 MB | 5.299 MB | 26.5887 ± 0.2359 ms | perf_counter | 26.5887 ± 0.2359 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.4842 | 52.68% |
| **ONNX FP32 (CPU)** | 2.239 MB | 6.964 MB | 14.9316 ± 0.3014 ms | perf_counter | 14.9316 ± 0.3014 ms | 0.8336 | 0.9468 | 0.7607 | 0.8436 | 0.8835 | 0.08% |
| **ONNX FP16 (CPU)** | 1.123 MB | 5.847 MB | 38.5714 ± 0.4502 ms | perf_counter | 38.5714 ± 0.4502 ms | 0.8407 | 0.9643 | 0.7580 | 0.8488 | 0.8844 | 2.68% |
| **ONNX INT8 (CPU)** | 2.052 MB | 6.777 MB | 15.1166 ± 0.2688 ms | perf_counter | 15.1166 ± 0.2688 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.7692 | 52.68% |
| **TensorRT FP16 (GPU)** | 9.278 MB | 14.002 MB | 19.9151 ± 0.9557 ms | cuda_event | 21.0181 ± 0.9716 ms | 0.8344 | 0.9483 | 0.7607 | 0.8442 | 0.8834 | 0.00% |
| **TensorRT FP32 (GPU)** | 9.278 MB | 14.002 MB | 19.9863 ± 0.9965 ms | cuda_event | 21.1038 ± 1.0190 ms | 0.8344 | 0.9483 | 0.7607 | 0.8442 | 0.8834 | 0.00% |
