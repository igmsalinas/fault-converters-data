# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.

| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 15.271 ± 0.088 ms | perf_counter | 15.271 ± 0.088 ms | 0.000 MB | 0.000 MB | 2463.500 MB | 10.613 MB |
| **Keras FP32 (CPU)** | 92.935 ± 1.112 ms | perf_counter | 92.935 ± 1.112 ms | 3.418 MB | 0.000 MB | 412.008 MB | 0.000 MB |
| **TFLite FP16** | 15.424 ± 0.204 ms | perf_counter | 15.424 ± 0.204 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |
| **TFLite INT8** | 14.731 ± 0.134 ms | perf_counter | 14.731 ± 0.134 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |
| **ONNX FP32 (CPU)** | 7.222 ± 0.061 ms | perf_counter | 7.222 ± 0.061 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |
| **ONNX FP16 (CPU)** | 9.703 ± 0.154 ms | perf_counter | 9.703 ± 0.154 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |
| **ONNX INT8 (CPU)** | 7.598 ± 0.110 ms | perf_counter | 7.598 ± 0.110 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |
| **TensorRT FP16 (GPU)** | 2.987 ± 0.049 ms | cuda_event | 3.984 ± 0.055 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |
| **TensorRT FP32 (GPU)** | 2.993 ± 0.049 ms | cuda_event | 4.012 ± 0.058 ms | 0.000 MB | 0.000 MB | 2463.520 MB | 10.613 MB |

## 2. Classification Metrics Evaluation

| Model Format | Graph Size (MB) | Detector Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFLite FP32** | 1.272 MB | 2.454 MB | 15.3112 ± 0.0815 ms | perf_counter | 15.3112 ± 0.0815 ms | 0.8707 | 0.9650 | 0.8102 | 0.8808 | 0.9100 | 0.00% |
| **Keras FP32 (CPU)** | 1.214 MB | 2.395 MB | 93.5164 ± 3.7773 ms | perf_counter | 93.5164 ± 3.7773 ms | 0.8707 | 0.9650 | 0.8102 | 0.8808 | 0.9100 | 0.00% |
| **TFLite FP16** | 0.669 MB | 1.850 MB | 15.3077 ± 0.0869 ms | perf_counter | 15.3077 ± 0.0869 ms | 0.8636 | 0.9285 | 0.8329 | 0.8781 | 0.9106 | 3.39% |
| **TFLite INT8** | 0.375 MB | 1.556 MB | 14.6792 ± 0.0950 ms | perf_counter | 14.6792 ± 0.0950 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.6924 | 50.47% |
| **ONNX FP32 (CPU)** | 1.229 MB | 2.410 MB | 7.1585 ± 0.0479 ms | perf_counter | 7.1585 ± 0.0479 ms | 0.8707 | 0.9650 | 0.8102 | 0.8808 | 0.9100 | 0.00% |
| **ONNX FP16 (CPU)** | 0.642 MB | 1.823 MB | 9.6901 ± 0.0808 ms | perf_counter | 9.6901 ± 0.0808 ms | 0.8762 | 0.9610 | 0.8235 | 0.8870 | 0.9130 | 1.50% |
| **ONNX INT8 (CPU)** | 0.480 MB | 1.662 MB | 7.5847 ± 0.1058 ms | perf_counter | 7.5847 ± 0.1058 ms | 0.5899 | 0.5899 | 1.0000 | 0.7421 | 0.1786 | 50.47% |
| **TensorRT FP16 (GPU)** | 0.960 MB | 2.141 MB | 2.9800 ± 0.0363 ms | cuda_event | 3.9724 ± 0.0424 ms | 0.7413 | 0.7188 | 0.9225 | 0.8080 | 0.9130 | 26.18% |
| **TensorRT FP32 (GPU)** | 1.563 MB | 2.745 MB | 2.9608 ± 0.0386 ms | cuda_event | 3.9885 ± 0.0552 ms | 0.8707 | 0.9650 | 0.8102 | 0.8808 | 0.9100 | 0.00% |
