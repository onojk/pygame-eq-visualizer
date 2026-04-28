import glfw
from OpenGL.GL import glClear, GL_COLOR_BUFFER_BIT
from OpenGL.GL import glGetString, GL_VERSION

def main():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")

    window = glfw.create_window(800, 600, "GLFW OpenGL Context", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window.")

    glfw.make_context_current(window)
    print("OpenGL Version:", glGetString(GL_VERSION).decode())

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

