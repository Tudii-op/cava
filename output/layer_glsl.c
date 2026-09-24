// Layer-shell based replacement for sdl_glsl.c: renders the same GLSL
// visualizers, but as a wlr-layer-shell surface instead of a regular
// SDL/xdg-toplevel window. Layer-shell surfaces are never decorated
// (no border/shadow/titlebar), never focus-managed, and can request a
// real alpha-capable EGL surface, so the compositor shows the desktop
// through anything the shader doesn't paint -- exactly how a terminal's
// "background_opacity" works, just via a different Wayland protocol.
#define GL_GLEXT_PROTOTYPES 0
#include <GL/gl.h>
#include <GL/glext.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <wayland-client.h>
#include <wayland-egl.h>

#include "sdl_glsl.h"
#include "wlr-layer-shell-unstable-v1-client-protocol.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../util.h"

static struct wl_display *display = NULL;
static struct wl_registry *registry = NULL;
static struct wl_compositor *compositor = NULL;
static struct zwlr_layer_shell_v1 *layer_shell = NULL;
static struct wl_surface *surface = NULL;
static struct zwlr_layer_surface_v1 *layer_surface = NULL;
static struct wl_egl_window *egl_window = NULL;

static EGLDisplay egl_display = EGL_NO_DISPLAY;
static EGLContext egl_context = EGL_NO_CONTEXT;
static EGLSurface egl_surface = EGL_NO_SURFACE;

static int surf_width, surf_height;
static bool configured = false;
static bool closed = false;

GLuint shading_program;
GLint uniform_bars;
GLint uniform_previous_bars;
GLint uniform_bars_count;
GLint uniform_time;
GLint uniform_input_texture;
GLuint fbo;
GLuint texture;
struct timespec start_time;

struct colors {
    uint16_t R;
    uint16_t G;
    uint16_t B;
};

static void parse_color(char *color_string, struct colors *color) {
    if (color_string[0] == '#') {
        sscanf(++color_string, "%02hx%02hx%02hx", &color->R, &color->G, &color->B);
    }
}

GLuint get_shader(GLenum, const char *);
GLuint custom_shaders(const char *, const char *);
const char *read_file(const char *);
GLuint compile_shader(GLenum type, const char **);
GLuint program_check(GLuint);

// --- Wayland registry: bind compositor + layer-shell globals ---

static void registry_global(void *data, struct wl_registry *reg, uint32_t name,
                             const char *interface, uint32_t version) {
    (void)data;
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        compositor = wl_registry_bind(reg, name, &wl_compositor_interface, 4);
    } else if (strcmp(interface, zwlr_layer_shell_v1_interface.name) == 0) {
        layer_shell = wl_registry_bind(reg, name, &zwlr_layer_shell_v1_interface, 1);
    }
}

static void registry_global_remove(void *data, struct wl_registry *reg, uint32_t name) {
    (void)data;
    (void)reg;
    (void)name;
}

static const struct wl_registry_listener registry_listener = {
    .global = registry_global,
    .global_remove = registry_global_remove,
};

// --- layer_surface events ---

static void layer_surface_configure(void *data, struct zwlr_layer_surface_v1 *ls, uint32_t serial,
                                     uint32_t width, uint32_t height) {
    (void)data;
    (void)width;
    (void)height;
    zwlr_layer_surface_v1_ack_configure(ls, serial);
    configured = true;
}

static void layer_surface_closed(void *data, struct zwlr_layer_surface_v1 *ls) {
    (void)data;
    (void)ls;
    closed = true;
}

static const struct zwlr_layer_surface_v1_listener layer_surface_listener = {
    .configure = layer_surface_configure,
    .closed = layer_surface_closed,
};

void init_sdl_glsl_window(int width, int height, int x, int y, int full_screen,
                          char *const vertex_shader, char *const fragmnet_shader) {
    (void)full_screen;

    surf_width = width;
    surf_height = height;

    display = wl_display_connect(NULL);
    if (!display) {
        fprintf(stderr, "could not connect to Wayland display\n");
        exit(1);
    }

    registry = wl_display_get_registry(display);
    wl_registry_add_listener(registry, &registry_listener, NULL);
    wl_display_roundtrip(display);

    if (!compositor || !layer_shell) {
        fprintf(stderr, "compositor does not support wlr-layer-shell\n");
        exit(1);
    }

    surface = wl_compositor_create_surface(compositor);
    layer_surface = zwlr_layer_shell_v1_get_layer_surface(
        layer_shell, surface, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BOTTOM, "cava");
    zwlr_layer_surface_v1_add_listener(layer_surface, &layer_surface_listener, NULL);

    zwlr_layer_surface_v1_set_size(layer_surface, width, height);
    zwlr_layer_surface_v1_set_keyboard_interactivity(
        layer_surface, ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);

    if (x < 0 || y < 0) {
        // No anchor: the compositor centers the surface on the output.
        zwlr_layer_surface_v1_set_anchor(layer_surface, 0);
    } else {
        // Anchor top-left and use margins to place it at an exact pixel
        // position, the same trick kitty's layer-shell panel uses.
        zwlr_layer_surface_v1_set_anchor(layer_surface, ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
                                                             ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT);
        zwlr_layer_surface_v1_set_margin(layer_surface, y, 0, 0, x);
    }

    wl_surface_commit(surface);
    while (!configured && !closed) {
        wl_display_dispatch(display);
    }
    if (closed) {
        fprintf(stderr, "layer surface closed before it was configured\n");
        exit(1);
    }

    egl_window = wl_egl_window_create(surface, width, height);

    egl_display = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, display, NULL);
    if (egl_display == EGL_NO_DISPLAY) {
        // Fallback for EGL implementations without eglGetPlatformDisplay.
        egl_display = eglGetDisplay((EGLNativeDisplayType)display);
    }
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "could not get EGL display\n");
        exit(1);
    }
    EGLint egl_major, egl_minor;
    if (!eglInitialize(egl_display, &egl_major, &egl_minor)) {
        fprintf(stderr, "could not initialize EGL\n");
        exit(1);
    }
    eglBindAPI(EGL_OPENGL_API);

    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE,
    };
    EGLConfig egl_config;
    EGLint num_config;
    if (!eglChooseConfig(egl_display, config_attribs, &egl_config, 1, &num_config) ||
        num_config < 1) {
        fprintf(stderr, "could not choose an alpha-capable EGL config\n");
        exit(1);
    }

    egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, NULL);
    if (egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "could not create EGL context\n");
        exit(1);
    }

    egl_surface =
        eglCreateWindowSurface(egl_display, egl_config, (EGLNativeWindowType)egl_window, NULL);
    if (egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "could not create EGL surface\n");
        exit(1);
    }

    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) {
        fprintf(stderr, "could not activate EGL context\n");
        exit(1);
    }
    eglSwapInterval(egl_display, 1);
    glViewport(0, 0, width, height);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    shading_program = custom_shaders(vertex_shader, fragmnet_shader);
    glReleaseShaderCompiler();
    if (shading_program == 0) {
        fprintf(stderr, "could not compile shaders\n");
        exit(1);
    }

    glUseProgram(shading_program);

    GLint gVertexPos2DLocation = glGetAttribLocation(shading_program, "vertexPosition_modelspace");
    if (gVertexPos2DLocation == -1) {
        fprintf(stderr, "could not find vertex position shader variable!\n");
        exit(1);
    }

    // Transparent clear: only what the shader explicitly paints (the bars)
    // is opaque; everywhere else stays alpha=0 and shows the desktop.
    glClearColor(0.f, 0.f, 0.f, 0.f);

    GLfloat vertexData[] = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
    GLuint indexData[] = {0, 1, 2, 3};

    GLuint gVBO = 0;
    glGenBuffers(1, &gVBO);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBufferData(GL_ARRAY_BUFFER, 2 * 4 * sizeof(GLfloat), vertexData, GL_STATIC_DRAW);

    GLuint gIBO = 0;
    glGenBuffers(1, &gIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLuint), indexData, GL_STATIC_DRAW);

    GLuint gVAO = 0;
    glGenVertexArrays(1, &gVAO);
    glBindVertexArray(gVAO);
    glEnableVertexAttribArray(gVertexPos2DLocation);

    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glVertexAttribPointer(gVertexPos2DLocation, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIBO);

    uniform_bars = glGetUniformLocation(shading_program, "bars");
    uniform_previous_bars = glGetUniformLocation(shading_program, "previous_bars");
    uniform_bars_count = glGetUniformLocation(shading_program, "bars_count");
    uniform_time = glGetUniformLocation(shading_program, "shader_time");
    uniform_input_texture = glGetUniformLocation(shading_program, "inputTexture");

    if (uniform_input_texture != -1) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "Framebuffer not complete!\n");
            exit(1);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    int error = glGetError();
    if (error != 0) {
        fprintf(stderr, "glError on init: %d\n", error);
        exit(1);
    }
}

void init_sdl_glsl_surface(int *w, int *h, char *const fg_color_string, char *const bg_color_string,
                           int bar_width, int bar_spacing, int gradient, int gradient_count,
                           char **gradient_color_strings) {
    struct colors color = {0};

    GLint uniform_bg_col = glGetUniformLocation(shading_program, "bg_color");
    parse_color(bg_color_string, &color);
    glUniform3f(uniform_bg_col, (float)color.R / 255.0, (float)color.G / 255.0,
                (float)color.B / 255.0);

    GLint uniform_fg_col = glGetUniformLocation(shading_program, "fg_color");
    parse_color(fg_color_string, &color);
    glUniform3f(uniform_fg_col, (float)color.R / 255.0, (float)color.G / 255.0,
                (float)color.B / 255.0);

    GLint uniform_res = glGetUniformLocation(shading_program, "u_resolution");
    *w = surf_width;
    *h = surf_height;
    glUniform3f(uniform_res, (float)surf_width, (float)surf_height, 0.0f);

    GLint uniform_bar_width = glGetUniformLocation(shading_program, "bar_width");
    glUniform1i(uniform_bar_width, bar_width);

    GLint uniform_bar_spacing = glGetUniformLocation(shading_program, "bar_spacing");
    glUniform1i(uniform_bar_spacing, bar_spacing);

    GLint uniform_gradient_count = glGetUniformLocation(shading_program, "gradient_count");
    if (gradient == 0)
        gradient_count = 0;
    glUniform1i(uniform_gradient_count, gradient_count);

    GLint uniform_gradient_colors = glGetUniformLocation(shading_program, "gradient_colors");
    float gradient_colors[8][3] = {0};
    for (int i = 0; i < gradient_count; i++) {
        parse_color(gradient_color_strings[i], &color);
        gradient_colors[i][0] = (float)color.R / 255.0;
        gradient_colors[i][1] = (float)color.G / 255.0;
        gradient_colors[i][2] = (float)color.B / 255.0;
    }
    glUniform3fv(uniform_gradient_colors, 8, (const GLfloat *)gradient_colors);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_INT, NULL);
    eglSwapBuffers(egl_display, egl_surface);
}

int draw_sdl_glsl(int bars_count, const float bars[], const float previous_bars[], int frame_time,
                  int re_paint, int continuous_rendering) {
    int rc = 0;

    if (re_paint || continuous_rendering) {
        if (uniform_input_texture != -1) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glUniform1i(uniform_input_texture, 0);
        }
        if (uniform_bars != -1)
            glUniform1fv(uniform_bars, bars_count, bars);
        if (uniform_previous_bars != -1)
            glUniform1fv(uniform_previous_bars, bars_count, previous_bars);
        if (uniform_bars_count != -1)
            glUniform1i(uniform_bars_count, bars_count);
        if (uniform_time != -1) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (double)(now.tv_sec - start_time.tv_sec) +
                              (double)(now.tv_nsec - start_time.tv_nsec) / 1e9;
            float wrapped = (float)fmod(elapsed, 4096.0);
            glUniform1f(uniform_time, wrapped);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_INT, NULL);
        eglSwapBuffers(egl_display, egl_surface);

        if (uniform_input_texture != -1) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_INT, NULL);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }

    struct timespec sleep_time = {.tv_sec = frame_time / 1000,
                                   .tv_nsec = (frame_time % 1000) * 1000000L};
    nanosleep(&sleep_time, NULL);

    wl_display_flush(display);
    while (wl_display_prepare_read(display) != 0) {
        wl_display_dispatch_pending(display);
    }
    wl_display_read_events(display);
    wl_display_dispatch_pending(display);

    if (closed)
        rc = -2;

    return rc;
}

// general: cleanup
void cleanup_sdl_glsl(void) {
    if (fbo != 0) {
        glDeleteFramebuffers(1, &fbo);
        fbo = 0;
    }
    if (texture != 0) {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
    if (egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_surface != EGL_NO_SURFACE)
            eglDestroySurface(egl_display, egl_surface);
        if (egl_context != EGL_NO_CONTEXT)
            eglDestroyContext(egl_display, egl_context);
        eglTerminate(egl_display);
    }
    if (egl_window)
        wl_egl_window_destroy(egl_window);
    if (layer_surface)
        zwlr_layer_surface_v1_destroy(layer_surface);
    if (surface)
        wl_surface_destroy(surface);
    if (display)
        wl_display_disconnect(display);
}

const char *read_file(const char *filename) {
    long length = 0;
    char *result = NULL;
    FILE *file = fopen(filename, "r");
    if (file) {
        int status = fseek(file, 0, SEEK_END);
        if (status != 0) {
            fclose(file);
            return NULL;
        }
        length = ftell(file);
        status = fseek(file, 0, SEEK_SET);
        if (status != 0) {
            fclose(file);
            return NULL;
        }
        result = malloc((length + 1) * sizeof(char));
        if (result) {
            size_t actual_length = fread(result, sizeof(char), length, file);
            result[actual_length++] = '\0';
        }
        fclose(file);
        return result;
    }
    fprintf(stderr, "Couldn't open shader file %s", filename);
    exit(1);
    return NULL;
}

GLuint custom_shaders(const char *vsPath, const char *fsPath) {
    GLuint vertexShader = get_shader(GL_VERTEX_SHADER, vsPath);
    GLuint fragmentShader = get_shader(GL_FRAGMENT_SHADER, fsPath);

    shading_program = glCreateProgram();

    glAttachShader(shading_program, vertexShader);
    glAttachShader(shading_program, fragmentShader);

    glLinkProgram(shading_program);

    GLuint status = program_check(shading_program);
    if (status == GL_FALSE)
        return 0;
    return shading_program;
}

GLuint get_shader(GLenum eShaderType, const char *filename) {
    const char *shaderSource = read_file(filename);
    GLuint shader = compile_shader(eShaderType, &shaderSource);
    free((char *)shaderSource);
    return shader;
}

GLuint compile_shader(GLenum type, const char **sources) {
    GLuint shader;
    GLint success, len;
    GLsizei srclens[1];

    srclens[0] = (GLsizei)strlen(sources[0]);

    shader = glCreateShader(type);
    glShaderSource(shader, 1, sources, srclens);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        if (len > 1) {
            char *log = malloc(len);
            glGetShaderInfoLog(shader, len, NULL, log);
            fprintf(stderr, "%s\n\n", log);
            free(log);
        }
        fprintf(stderr, "Error compiling shader.\n");
        exit(1);
    }
    return shader;
}

GLuint program_check(GLuint program) {
    GLint status;
    glValidateProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        GLint len;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        if (len > 1) {
            char *log = malloc(len);
            glGetProgramInfoLog(program, len, &len, log);
            fprintf(stderr, "%s\n\n", log);
            free(log);
        }
        fprintf(stderr, "Error linking shader default program.\n");
        return GL_FALSE;
    }
    return GL_TRUE;
}
