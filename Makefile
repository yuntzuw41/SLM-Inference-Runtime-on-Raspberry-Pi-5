# ECPS 205 Final Project build file (Raspberry Pi 5, AArch64, GCC 14.2.0)
# Usage (from this folder):   make all
# run_neon.c uses <arm_neon.h>, so Task 4 and Task 5 must be built on the Pi.

CC = gcc

.PHONY: all task2 task3 task4 task5 profile link clean

all: task2 task3 task4 task5

# Task 2: native compilation test
task2: task2_pi_setup/hello
task2_pi_setup/hello: task2_pi_setup/hello.c
	$(CC) -Wall -O2 -o $@ $<

# Task 3: unmodified llama2.c runtime (Scalar baseline)
task3: task3_baseline/run_scalar
task3_baseline/run_scalar: task3_baseline/run.c
	$(CC) -O3 -o $@ $< -lm

# Task 4: NEON runtime, timing builds, correctness test and kernel benchmark.
# run_scalar is built here too, so benchmark_runtime.sh can compare both runtimes.
task4: task4_neon/run_scalar task4_neon/run_neon task4_neon/run_scalar_timing \
       task4_neon/run_neon_timing task4_neon/test_neon task4_neon/benchmark_neon
task4_neon/run_scalar: task3_baseline/run.c
	$(CC) -O3 -o $@ $< -lm
task4_neon/run_neon: task4_neon/run_neon.c
	$(CC) -O3 -o $@ $< -lm
task4_neon/run_scalar_timing: task4_neon/run_scalar_timing.c
	$(CC) -O3 -o $@ $< -lm
task4_neon/run_neon_timing: task4_neon/run_neon_timing.c
	$(CC) -O3 -o $@ $< -lm
task4_neon/test_neon: task4_neon/test_neon.c
	$(CC) -O2 -Wall -o $@ $< -lm
# auto-vectorization disabled so the scalar kernel stays scalar
task4_neon/benchmark_neon: task4_neon/benchmark_neon.c
	$(CC) -O2 -Wall -fno-tree-vectorize -o $@ $< -lm

# gprof builds (run them from a separate directory: gprof writes gmon.out to the cwd)
profile: task3_baseline/run.c
	$(CC) -O2 -pg -o task4_neon/run_profile $< -lm
	$(CC) -O2 -pg -fno-inline -o task4_neon/run_profile_noinline $< -lm

# Task 5: camera application
task5: task5_camera_app/camera_story task5_camera_app/image_analysis link
task5_camera_app/camera_story: task5_camera_app/camera_story.c
	$(CC) -O2 -Wall -Wextra -o $@ $<
task5_camera_app/image_analysis: task5_camera_app/image_analysis.c
	$(CC) -O2 -Wall -Wextra -o $@ $< -lm

# camera_story looks for the runtime in ../llama2.c, so point that name at task4_neon
link:
	@test -e llama2.c || ln -s task4_neon llama2.c

clean:
	rm -f task2_pi_setup/hello task3_baseline/run_scalar \
	      task4_neon/run_scalar task4_neon/run_neon task4_neon/run_scalar_timing \
	      task4_neon/run_neon_timing task4_neon/test_neon task4_neon/benchmark_neon \
	      task4_neon/run_profile task4_neon/run_profile_noinline \
	      task5_camera_app/camera_story task5_camera_app/image_analysis
