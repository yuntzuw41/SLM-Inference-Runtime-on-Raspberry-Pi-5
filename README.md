# ECPS 205 Final Project: SLM Inference Runtime on Raspberry Pi 5

Team: Yuntzu Wang, Morris Chen, Janice Lin

TinyStories 15M (Karpathy's llama2.c) running natively in C on a Raspberry Pi 5, accelerated with Arm NEON, and extended into a camera-based storytelling application.

## Folder layout

```
ECPS205_Final_Submission/
├── Makefile               builds every task (run on the Pi)
├── task2_pi_setup/        hello.c native compilation test
├── task3_baseline/        unmodified llama2.c run.c (Scalar baseline)
├── task4_neon/            NEON runtime, timing builds, tests, benchmarks, results
├── task5_camera_app/      camera_story.c, image_analysis.c, recorded captures
└── figures/               report figures and the scripts that draw them
```

Task 1 ran the unmodified llama2.c on a laptop (WSL Ubuntu, `make run`). It has no code of our own; `task3_baseline/run.c` is the same file.

| Task | Files | Evidence |
|---|---|---|
| 2 | `hello.c` | `evidence/task2_environment.txt`, `evidence/task2_hello.txt` |
| 3 | `run.c` (original llama2.c) | `evidence/task3_native_inference.txt` |
| 4 | `run_neon.c` (only `matmul()` changed), `run_scalar_timing.c`, `run_neon_timing.c`, `test_neon.c`, `benchmark_neon.c`, `benchmark_runtime.sh` | `results/` (5 alternating runs), `evidence/` (gprof, kernel benchmark, timing, memory, disassembly) |
| 5 | `camera_story.c`, `image_analysis.c` | `captures/capture_NNN/` (image, colors, prompt, story, metrics) |

## Platform

Raspberry Pi 5 Model B (4 GB), Cortex-A76 ×4 at 2.4 GHz, Debian 13 (trixie), GCC 14.2.0, OV5647 camera on CAM/DISP 0. All runtimes are single-threaded.

## Setup

```bash
sudo apt install build-essential ffmpeg rpicam-apps
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
cp stories15M.bin task3_baseline/
mv stories15M.bin task4_neon/
make all
```

`stories15M.bin` (58 MB) is not included; download it as above. `tokenizer.bin` is included in `task3_baseline/` and `task4_neon/`.

## Reproducing the results

Task 2
```bash
./task2_pi_setup/hello
```

Task 3 (from `task3_baseline/`)
```bash
./run_scalar stories15M.bin -t 0 -n 128 -s 42 -i "Once upon a time, there was a little robot named Alex."
```

Task 4 (from `task4_neon/`)
```bash
./test_neon                     # small-matrix correctness check
./benchmark_neon                # 2048x768 MatVec, 300 iterations: speedup and max error
bash benchmark_runtime.sh       # 5 alternating Scalar/NEON runs -> results/runtime_benchmark.csv
P="Once upon a time, there was a little robot named Alex."
./run_scalar_timing stories15M.bin -t 0 -n 128 -s 42 -i "$P" > /dev/null   # TTFT / TPOT
./run_neon_timing   stories15M.bin -t 0 -n 128 -s 42 -i "$P" > /dev/null
/usr/bin/time -v ./run_neon stories15M.bin -t 0 -n 128 -s 42 -i "$P" > /dev/null   # peak RSS
python3 plot_figure10.py; python3 plot_figure13.py
```

Profiling (from a separate directory, because gprof writes `gmon.out` to the current directory)
```bash
mkdir -p ../profile_run && cd ../profile_run
gcc -O2 -pg -fno-inline -o prof ../task3_baseline/run.c -lm
./prof ../task4_neon/stories15M.bin -z ../task4_neon/tokenizer.bin -t 0 -n 256 -s 42 -i "$P" > /dev/null
gprof -b -p prof gmon.out | head -15
```

Task 5 (from `task5_camera_app/`). `camera_story` looks for the runtime in `../llama2.c`; `make all` creates a link named `llama2.c` that points to `task4_neon/`.
```bash
rpicam-hello --list-cameras                            # check the camera first
./camera_story --runtime neon                          # live capture
./camera_story --image example_image.jpg --runtime neon   # without a camera
```
Each run creates `captures/capture_NNN/` with `image.jpg`, `image.ppm`, `colors.txt`, `prompt.txt`, `story.txt`, `inference.log` and `metrics.txt`. The `image.ppm` files are left out of this submission because they are regenerated from `image.jpg`. Captures 002 and 003 were empty test folders and are not included.

Figures (from `figures/`)
```bash
python3 plot_figure1.py              # Figure 1, inference pipeline
python3 plot_figures_14_17_19.py     # Figures 14, 17, 19 (reads ../task5_camera_app/captures)
```

## Key results

| Measurement | Scalar | NEON |
|---|---|---|
| Median throughput, 5 runs (tok/s) | 77.82 | 202.88 (2.61×) |
| TTFT / TPOT (median of 5) | 178.7 ms / 12.86 ms | 68.5 ms / 4.92 ms |
| Peak RSS | 64,592 KB | 64,592 KB |
| Standalone 2048×768 MatVec | 1.311 ms | 0.330 ms (3.975×) |
| Share of profiled time in `matmul()` (`-fno-inline`) | 95.53% | |

Task 5, live camera, NEON runtime: 3.1 to 3.6 s end to end, of which about 0.66 s is story generation.

## Known limitations

- Single-threaded only; no OpenMP.
- Only the Sunny Sky environment rule has been triggered by a recorded image (offline test, Capture 001).
- Task 5 has no controlled Scalar-versus-NEON comparison of the complete application.
- The standalone MatVec kernel uses a single accumulator, so it is limited by the 2-cycle FMA accumulate latency of the Cortex-A76 (2.4 / 9.6 GFLOP/s limits); multiple accumulators were not tried.
- The memory bandwidth ceiling of the Pi 5 was not measured.

## Acknowledgement

Based on [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy (MIT License). `run.c` and `tokenizer.bin` are from that repository.
