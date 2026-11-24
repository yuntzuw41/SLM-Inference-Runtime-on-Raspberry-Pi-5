#define _XOPEN_SOURCE 700
/* ECPS 205 Task 5: Offline camera / saved-image to story pipeline.
 * Usage: ./camera_story [--image /path/to/photo.jpg] [--runtime neon|scalar]
 * Builds on the existing image_analysis.c and llama2.c runtime.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#define MAX_CAPTURES 999999

static double clock_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { perror("clock_gettime"); exit(1); }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int path_join(char *dst, size_t size, const char *a, const char *b) {
    int n = snprintf(dst, size, "%s/%s", a, b);
    if (n < 0 || (size_t)n >= size) {
        fprintf(stderr, "Path is too long: %s/%s\n", a, b);
        return 0;
    }
    return 1;
}

static int copy_file(const char *source, const char *destination) {
    int in = open(source, O_RDONLY);
    if (in < 0) { perror("open source image"); return 0; }
    int out = open(destination, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (out < 0) { perror("create image copy"); close(in); return 0; }
    char buffer[32768];
    ssize_t n;
    int ok = 1;
    while ((n = read(in, buffer, sizeof(buffer))) > 0) {
        ssize_t off = 0;
        while (off < n) {
            ssize_t written = write(out, buffer + off, (size_t)(n - off));
            if (written <= 0) { perror("write image"); ok = 0; break; }
            off += written;
        }
        if (!ok) break;
    }
    if (n < 0) { perror("read image"); ok = 0; }
    if (close(in) != 0) ok = 0;
    if (close(out) != 0) ok = 0;
    return ok;
}

/* If a log path is NULL, inherit the respective terminal stream. */
static int run_program(const char *working_dir, const char *stdout_file,
                       const char *stderr_file, char *const args[]) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 0; }
    if (pid == 0) {
        if (chdir(working_dir) != 0) { perror("chdir"); _exit(127); }
        const char *files[2] = { stdout_file, stderr_file };
        const int targets[2] = { STDOUT_FILENO, STDERR_FILENO };
        for (int i = 0; i < 2; ++i) {
            if (files[i] == NULL) continue;
            int fd = open(files[i], O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0 || dup2(fd, targets[i]) < 0) {
                perror("redirect output"); _exit(127);
            }
            close(fd);
        }
        execvp(args[0], args);
        perror("execvp");
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return 0; }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "Command failed: %s (exit status: %d)\n", args[0],
                WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 0;
    }
    return 1;
}

static void print_file(const char *file) {
    FILE *fp = fopen(file, "r");
    if (!fp) { perror(file); return; }
    char line[512];
    while (fgets(line, sizeof(line), fp)) fputs(line, stdout);
    fclose(fp);
}

static void usage(const char *arg0) {
    printf("Usage: %s [--image IMAGE.jpg] [--runtime neon|scalar]\n", arg0);
    puts("  No --image: capture a new image from rpicam-still.");
    puts("  --image: test the pipeline using an existing JPEG (camera not required).");
    puts("  --runtime: neon (default) or scalar; both must be built in ../llama2.c.");
}

int main(int argc, char **argv) {
    const char *input_image = NULL;
    const char *runtime = "neon";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        if (!strcmp(argv[i], "--image") && i + 1 < argc) { input_image = argv[++i]; continue; }
        if (!strcmp(argv[i], "--runtime") && i + 1 < argc) { runtime = argv[++i]; continue; }
        usage(argv[0]); return 2;
    }
    if (strcmp(runtime, "neon") && strcmp(runtime, "scalar")) {
        fprintf(stderr, "Invalid runtime: %s\n", runtime); return 2;
    }
    char input_absolute[PATH_MAX];
    if (input_image) {
        if (!realpath(input_image, input_absolute)) {
            perror("Cannot resolve --image path"); return 1;
        }
        input_image = input_absolute;
    }
    char task_dir[PATH_MAX], llama_dir[PATH_MAX], analysis_path[PATH_MAX];
    if (!getcwd(task_dir, sizeof(task_dir))) { perror("getcwd"); return 1; }
    if (!path_join(llama_dir, sizeof(llama_dir), task_dir, "../llama2.c") ||
        !path_join(analysis_path, sizeof(analysis_path), task_dir, "image_analysis")) return 1;
    char capture_root[PATH_MAX];
    if (!path_join(capture_root, sizeof(capture_root), task_dir, "captures")) return 1;
    if (mkdir(capture_root, 0755) != 0 && errno != EEXIST) { perror("mkdir captures"); return 1; }
    char capture_dir[PATH_MAX] = "";
    int capture_id = 0;
    for (int i = 1; i <= MAX_CAPTURES; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "capture_%03d", i);
        if (!path_join(capture_dir, sizeof(capture_dir), capture_root, name)) return 1;
        if (mkdir(capture_dir, 0755) == 0) { capture_id = i; break; }
        if (errno != EEXIST) { perror("mkdir capture"); return 1; }
    }
    if (!capture_id) { fputs("No available capture ID\n", stderr); return 1; }
    char image_path[PATH_MAX], colors_path[PATH_MAX], prompt_path[PATH_MAX];
    char story_path[PATH_MAX], inference_path[PATH_MAX], metrics_path[PATH_MAX];
    if (!path_join(image_path, sizeof(image_path), capture_dir, "image.jpg") ||
        !path_join(colors_path, sizeof(colors_path), capture_dir, "colors.txt") ||
        !path_join(prompt_path, sizeof(prompt_path), capture_dir, "prompt.txt") ||
        !path_join(story_path, sizeof(story_path), capture_dir, "story.txt") ||
        !path_join(inference_path, sizeof(inference_path), capture_dir, "inference.log") ||
        !path_join(metrics_path, sizeof(metrics_path), capture_dir, "metrics.txt")) return 1;

    printf("\n========== Camera Story Generator V3 ==========\n");
    printf("Capture ID: %03d\nDirectory: %s\nRuntime: %s\n", capture_id, capture_dir, runtime);
    fflush(stdout);
    double total_start = clock_ms();
    double capture_start = clock_ms();
    if (input_image) {
        printf("[1/5] Copying existing image (offline camera test)...\n");
        if (!copy_file(input_image, image_path)) return 1;
    } else {
        printf("[1/5] Capturing image...\n"); fflush(stdout);
        char *camera_args[] = {"rpicam-still", "-n", "-t", "2000", "--width", "640",
                               "--height", "480", "-o", "image.jpg", NULL};
        if (!run_program(capture_dir, NULL, NULL, camera_args)) return 1;
    }
    double capture_ms = clock_ms() - capture_start;

    printf("[2/5] Converting JPEG to RGB PPM...\n"); fflush(stdout);
    double conversion_start = clock_ms();
    char *ffmpeg_args[] = {"ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                           "-i", "image.jpg", "-frames:v", "1", "-pix_fmt", "rgb24",
                           "-update", "1", "image.ppm", NULL};
    if (!run_program(capture_dir, NULL, NULL, ffmpeg_args)) return 1;
    double conversion_ms = clock_ms() - conversion_start;

    printf("[3/5] Analyzing image colors...\n"); fflush(stdout);
    double analysis_start = clock_ms();
    char *analysis_args[] = {analysis_path, "image.ppm", NULL};
    if (!run_program(capture_dir, "colors.txt", NULL, analysis_args)) return 1;
    double analysis_ms = clock_ms() - analysis_start;
    print_file(colors_path);

    printf("\n[4/5] Reading generated prompt...\n");
    FILE *prompt_fp = fopen(prompt_path, "r");
    if (!prompt_fp) { perror("Cannot open prompt.txt"); return 1; }
    char prompt[512];
    if (!fgets(prompt, sizeof(prompt), prompt_fp)) {
        fputs("Cannot read prompt\n", stderr); fclose(prompt_fp); return 1;
    }
    fclose(prompt_fp);
    prompt[strcspn(prompt, "\r\n")] = '\0';
    if (!prompt[0]) { fputs("Empty prompt\n", stderr); return 1; }
    printf("Prompt: %s\n", prompt);

    printf("[5/5] Generating story using %s runtime...\n", runtime); fflush(stdout);
    const char *binary = strcmp(runtime, "neon") ? "./run_scalar" : "./run_neon";
    char *llama_args[] = {(char *)binary, "stories15M.bin", "-t", "0", "-n", "128",
                          "-s", "42", "-i", prompt, NULL};
    double generation_start = clock_ms();
    if (!run_program(llama_dir, story_path, inference_path, llama_args)) {
        fprintf(stderr, "Check runtime and model in %s; diagnostic log: %s\n", llama_dir, inference_path);
        return 1;
    }
    double generation_ms = clock_ms() - generation_start;
    double total_ms = clock_ms() - total_start;
    FILE *metrics = fopen(metrics_path, "w");
    if (!metrics) { perror("write metrics"); return 1; }
    fprintf(metrics, "runtime=%s\nimage_source=%s\ncapture_or_copy_ms=%.3f\nconversion_ms=%.3f\nanalysis_ms=%.3f\ngeneration_ms=%.3f\ntotal_ms=%.3f\n",
            runtime, input_image ? "existing_file" : "camera", capture_ms, conversion_ms,
            analysis_ms, generation_ms, total_ms);
    fclose(metrics);
    puts("\n========== Generated Story ==========");
    print_file(story_path);
    puts("\n========== Timing Summary (ms) ==========");
    print_file(metrics_path);
    printf("Saved to: %s\n", capture_dir);
    return 0;
}
