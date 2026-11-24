#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define NUM_COLORS 13
#define NUM_ENVIRONMENTS 10

// =====================================================
// Color Categories
// =====================================================

enum Color {
    RED,
    ORANGE,
    YELLOW,
    GREEN,
    CYAN,
    BLUE,
    PURPLE,
    PINK,
    BROWN,
    WHITE,
    GRAY,
    BLACK,
    OTHER
};

const char *color_names[NUM_COLORS] = {
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Cyan",
    "Blue",
    "Purple",
    "Pink",
    "Brown",
    "White",
    "Gray",
    "Black",
    "Other"
};

// =====================================================
// Data Structures
// =====================================================

typedef struct {
    double h;
    double s;
    double v;
} HSV;

typedef struct {
    const char *name;

    int color1;
    int color2;

    double min1;
    double min2;

    const char *prompt;

} EnvironmentRule;

// =====================================================
// Environment Rules
// =====================================================

const EnvironmentRule environment_rules[NUM_ENVIRONMENTS] = {

    {
        "Sunny Sky",
        BLUE, WHITE,
        30.0, 10.0,
        "Once upon a time, under a beautiful blue sky,"
    },

    {
        "Night Sky",
        BLACK, WHITE,
        35.0, 5.0,
        "Once upon a time, under a dark night sky,"
    },

    {
        "Forest",
        GREEN, BROWN,
        25.0, 8.0,
        "Once upon a time, deep inside a green forest,"
    },

    {
        "Ocean",
        BLUE, CYAN,
        30.0, 10.0,
        "Once upon a time, near the beautiful blue ocean,"
    },

    {
        "Desert",
        YELLOW, BROWN,
        20.0, 15.0,
        "Once upon a time, in a warm sandy desert,"
    },

    {
        "Snowy Landscape",
        WHITE, CYAN,
        40.0, 5.0,
        "Once upon a time, in a beautiful snowy land,"
    },

    {
        "Sunset",
        RED, ORANGE,
        20.0, 15.0,
        "Once upon a time, during a beautiful orange sunset,"
    },

    {
        "Flower Garden",
        GREEN, PINK,
        20.0, 10.0,
        "Once upon a time, in a garden full of flowers,"
    },

    {
        "City",
        GRAY, BLACK,
        25.0, 15.0,
        "Once upon a time, in a busy city,"
    },

    {
        "Grassland",
        GREEN, BLUE,
        35.0, 10.0,
        "Once upon a time, in a wide green grassland,"
    }

};

// =====================================================
// Color-based fallback prompts
// Used only if none of the 10 scene rules matches.
// Keep the image-derived input meaningful without inventing a scene.
// =====================================================

static const char *fallback_prompt_for_color(int color)
{
    static const char *const prompts[NUM_COLORS] = {
        "Once upon a time, in a land glowing with bright red colors,",
        "Once upon a time, in a world filled with warm orange colors,",
        "Once upon a time, in a land shining with golden yellow colors,",
        "Once upon a time, in a world filled with beautiful green colors,",
        "Once upon a time, in a magical place filled with cyan colors,",
        "Once upon a time, in a world painted in deep blue colors,",
        "Once upon a time, in a land glowing with purple colors,",
        "Once upon a time, in a magical world full of pink colors,",
        "Once upon a time, in a world painted in earthy brown colors,",
        "Once upon a time, in a bright world covered in white,",
        "Once upon a time, in a quiet world painted in shades of gray,",
        "Once upon a time, in a mysterious world covered in black,",
        "Once upon a time, in a colorful and mysterious place,"
    };
    if (color < 0 || color >= NUM_COLORS) return prompts[OTHER];
    return prompts[color];
}

// =====================================================
// RGB to HSV Conversion
// =====================================================

HSV rgb_to_hsv(unsigned char r,
               unsigned char g,
               unsigned char b)
{
    double rd = r / 255.0;
    double gd = g / 255.0;
    double bd = b / 255.0;

    double max = fmax(rd, fmax(gd, bd));
    double min = fmin(rd, fmin(gd, bd));
    double delta = max - min;

    HSV hsv = {0.0, 0.0, max};

    if (max > 0.0) {
        hsv.s = delta / max;
    }

    if (delta > 0.0) {

        if (max == rd) {
            hsv.h = 60.0 *
                    fmod((gd - bd) / delta, 6.0);
        }
        else if (max == gd) {
            hsv.h = 60.0 *
                    ((bd - rd) / delta + 2.0);
        }
        else {
            hsv.h = 60.0 *
                    ((rd - gd) / delta + 4.0);
        }

        if (hsv.h < 0.0) {
            hsv.h += 360.0;
        }
    }

    return hsv;
}

// =====================================================
// HSV Color Classification
// =====================================================

int classify_color(unsigned char r,
                   unsigned char g,
                   unsigned char b)
{
    HSV hsv = rgb_to_hsv(r, g, b);

    double h = hsv.h;
    double s = hsv.s;
    double v = hsv.v;

    // Black
    if (v < 0.18) {
        return BLACK;
    }

    // White and Gray
    if (s < 0.15) {

        if (v > 0.82) {
            return WHITE;
        }

        return GRAY;
    }

    // Low saturation
    if (s < 0.22) {
        return OTHER;
    }

    // Brown
    if (h >= 15.0 && h < 50.0 &&
        v >= 0.18 && v < 0.65) {

        return BROWN;
    }

    // Pink
    if ((h >= 330.0 || h < 15.0) &&
        s < 0.65 && v >= 0.65) {

        return PINK;
    }

    // Red
    if (h < 15.0 || h >= 345.0) {
        return RED;
    }

    // Orange
    else if (h < 45.0) {
        return ORANGE;
    }

    // Yellow
    else if (h < 70.0) {
        return YELLOW;
    }

    // Green
    else if (h < 165.0) {
        return GREEN;
    }

    // Cyan
    else if (h < 195.0) {
        return CYAN;
    }

    // Blue
    else if (h < 255.0) {
        return BLUE;
    }

    // Purple
    else if (h < 330.0) {
        return PURPLE;
    }

    // Pink
    else {
        return PINK;
    }
}

// =====================================================
// Read PPM Header Token
// =====================================================

int read_token(FILE *fp, char *buffer, size_t size)
{
    int c;
    size_t i = 0;

    do {
        c = fgetc(fp);

        if (c == '#') {

            while (c != '\n' && c != EOF) {
                c = fgetc(fp);
            }
        }

    } while (c == ' ' || c == '\n' ||
             c == '\r' || c == '\t');

    if (c == EOF) {
        return 0;
    }

    while (c != EOF &&
           c != ' ' &&
           c != '\n' &&
           c != '\r' &&
           c != '\t') {

        if (i + 1 >= size) {
            return 0;
        }

        buffer[i++] = (char)c;

        c = fgetc(fp);
    }

    buffer[i] = '\0';

    return i > 0;
}

// =====================================================
// Main Program
// =====================================================

int main(int argc, char *argv[])
{
    // -------------------------------------------------
    // Step 1: Open Image
    // -------------------------------------------------

    if (argc != 2) {

        fprintf(stderr,
                "Usage: %s <image.ppm>\n",
                argv[0]);

        return 1;
    }

    FILE *fp = fopen(argv[1], "rb");

    if (fp == NULL) {
        perror("Error opening image");
        return 1;
    }

    // -------------------------------------------------
    // Step 2: Read PPM Header
    // -------------------------------------------------

    char token[64];

    if (!read_token(fp, token, sizeof(token)) ||
        strcmp(token, "P6") != 0) {

        fprintf(stderr,
                "Error: Only binary PPM (P6) supported.\n");

        fclose(fp);
        return 1;
    }

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        return 1;
    }

    int width = atoi(token);

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        return 1;
    }

    int height = atoi(token);

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        return 1;
    }

    int maxval = atoi(token);

    if (width <= 0 ||
        height <= 0 ||
        maxval != 255 ||
        (long long)width * height > LONG_MAX / 3) {

        fprintf(stderr,
                "Invalid PPM dimensions or format.\n");

        fclose(fp);
        return 1;
    }

    // -------------------------------------------------
    // Step 3: Analyze Pixel Colors
    // -------------------------------------------------

    long total_pixels = (long)width * height;

    long counts[NUM_COLORS] = {0};

    unsigned char pixel[3];

    for (long i = 0; i < total_pixels; i++) {

        if (fread(pixel, 1, 3, fp) != 3) {

            fprintf(stderr,
                    "Error reading pixel data.\n");

            fclose(fp);
            return 1;
        }

        int color = classify_color(
            pixel[0],
            pixel[1],
            pixel[2]
        );

        counts[color]++;
    }

    fclose(fp);

    // -------------------------------------------------
    // Step 4: Calculate Color Percentages
    // -------------------------------------------------

    double percentages[NUM_COLORS];

    printf("\n========== HSV Color Analysis ==========\n");

    printf("Image: %s\n", argv[1]);

    printf("Resolution: %d x %d\n",
           width, height);

    printf("Total Pixels: %ld\n\n",
           total_pixels);

    int dominant = 0;

    for (int i = 0; i < NUM_COLORS; i++) {

        percentages[i] =
            (double)counts[i] /
            total_pixels * 100.0;

        printf("%-8s: %6.2f%%\n",
               color_names[i],
               percentages[i]);

        if (counts[i] > counts[dominant]) {
            dominant = i;
        }
    }

    printf("\nDominant Color: %s\n",
           color_names[dominant]);

    // -------------------------------------------------
    // Step 5: Find Dominant Recognized Color
    // -------------------------------------------------

    int dominant_recognized = 0;

    for (int i = 1; i < OTHER; i++) {

        if (counts[i] >
            counts[dominant_recognized]) {

            dominant_recognized = i;
        }
    }

    printf("Dominant Recognized Color: %s\n",
           color_names[dominant_recognized]);

    // -------------------------------------------------
    // Step 6: Multi-Color Environment Estimation
    // -------------------------------------------------

    const char *environment = "Unknown";

    const char *prompt =
        "Once upon a time, in a colorful and mysterious place,";

    double best_score = -1.0;

    printf("\n========== Environment Scores ==========\n");

    for (int i = 0; i < NUM_ENVIRONMENTS; i++) {

        EnvironmentRule rule = environment_rules[i];

        double p1 = percentages[rule.color1];
        double p2 = percentages[rule.color2];

        // Both colors must satisfy minimum thresholds
        if (p1 < rule.min1 || p2 < rule.min2) {

            printf("%-18s: Not matched\n",
                   rule.name);

            continue;
        }

        // Calculate environment score
        double score =
            0.6 * p1 + 0.4 * p2;

        printf("%-18s: %.2f\n",
               rule.name,
               score);

        // Select highest scoring environment
        if (score > best_score) {

            best_score = score;

            environment = rule.name;
            prompt = rule.prompt;
        }
    }

    printf("========================================\n");

    // When no rule matches, avoid giving all images the same story prompt.
    // Use the measured dominant color as a fallback instead of inventing
    // a semantic scene. OTHER uses the most common classified color if any.
    if (best_score < 0.0) {
        int fallback_color = dominant;
        if (fallback_color == OTHER) {
            long max_count = 0;
            fallback_color = OTHER;
            for (int i = 0; i < OTHER; i++) {
                if (counts[i] > max_count) {
                    max_count = counts[i];
                    fallback_color = i;
                }
            }
        }
        prompt = fallback_prompt_for_color(fallback_color);
        printf("Fallback based on dominant color: %s\n",
               color_names[fallback_color]);
    }

    // -------------------------------------------------
    // Step 7: Print Estimated Environment
    // -------------------------------------------------

    printf("\nEstimated Environment: %s\n",
           environment);

    if (best_score >= 0.0) {

        printf("Environment Score: %.2f\n",
               best_score);
    }
    else {

        printf("Environment Score: N/A\n");
    }

    // -------------------------------------------------
    // Step 8: Generate Story Prompt
    // -------------------------------------------------

    printf("\nGenerated Prompt:\n%s\n",
           prompt);

    // -------------------------------------------------
    // Step 9: Save Prompt
    // -------------------------------------------------

    FILE *prompt_file = fopen("prompt.txt", "w");

    if (prompt_file == NULL) {

        perror("Error creating prompt.txt");

        return 1;
    }

    if (fprintf(prompt_file, "%s\n", prompt) < 0) {
        perror("Error writing prompt.txt");
        fclose(prompt_file);
        return 1;
    }

    if (fclose(prompt_file) != 0) {
        perror("Error closing prompt.txt");
        return 1;
    }

    printf("\nPrompt saved to prompt.txt\n");

    printf("========================================\n");

    return 0;
}