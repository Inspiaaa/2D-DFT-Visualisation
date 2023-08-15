import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import math


INPUT_IMAGE_PATH = "input/stop.jpg"

# Resizes the input image to the target square size. Non-square images will be distorted.
IMAGE_SIZE = 256

# A constant with which the sinusoid brightness is multiplied.
BRIGHTNESS_FACTOR = 2

# A non-linear parameter that skews the brightness curve of the sinusoids.
# For values greater than 1 darker sinusoids (i.e. lower amplitude) appear brighter.
BRIGHTNESS_BIAS = 2

# Colormap used for displaying the 2D FFT.
COLORMAP = plt.get_cmap("viridis")

# Colormap used for pixels that have been "visited", i.e. added to the image so far.
VISITED_COLORMAP = plt.get_cmap("plasma")

# Methods to traverse / walk the frequency domain...
# ... by increasing Euclidean distance from the center, effectively resulting in increasing circle diameters.
#     It is slightly confusing as it jumps around the center because of the discrete frequency bins (pixels)
#     not lying perfectly on concentric circles.
ORDER_BY_EUCLIDEAN_DIST = lambda pos: pos[0] ** 2 + pos[1] ** 2

# ... by increasing Chebyshev distance from the center, effectively resulting in increasing sized squares.
#     It is a more intuitive way to traverse the frequency domain, as concentric squares lie exactly on
#     the frequency bins (pixels), thus avoiding the jumping around of the Euclidean distance.
#     The Chebyshev distance is slightly modified in order to mostly follow a perimeter tracing path.
ORDER_BY_CHEBYSHEV_DIST = lambda pos: max(abs(pos[0]), abs(pos[1]) + (0.1 if pos[1] > 0 else 0))

# Determines the traversal order (either ORDER_BY_EUCLIDEAN_DIST or ORDER_BY_CHEBYSHEV_DIST or custom).
SINUSOID_DRAW_ORDER = ORDER_BY_CHEBYSHEV_DIST


def compute_2d_complex_sinusoid(size, freq_x, freq_y, coefficient):
    values = np.arange(size)
    x, y = np.meshgrid(values, values)
    return coefficient * np.exp(2j * np.pi / size * (freq_x * x + freq_y * y)) / (size ** 2)


class Animator:
    def __init__(self, image_ax, layer_ax, fft_ax, fft):
        self.image_ax = image_ax
        self.layer_ax = layer_ax
        self.fft_ax = fft_ax

        self.fft = fft
        self.size = len(fft)

        # The image that is created iteratively.
        self.image = np.zeros((self.size, self.size), dtype=np.float)
        self.image_imshow = image_ax.imshow(self.image, cmap="gray", vmin=0, vmax=255)

        # Each layer that is added to the image.
        self.layer_imshow = layer_ax.imshow(np.zeros((self.size, self.size), dtype=np.float), cmap="gray")

        # The FFT.
        self.normalized_fft_image = np.log10(np.abs(self.fft) + 1)
        self.normalized_fft_image /= np.max(self.normalized_fft_image)
        self.normalized_fft_image = np.fft.fftshift(self.normalized_fft_image)

        self.visited_fft_image = COLORMAP(self.normalized_fft_image)[:, :, :3]

        self.fft_ax.set_title("FFT")
        self.fft_imshow = fft_ax.imshow(self.image)

        # Circle to highlight area on FFT to show location of current sinusoid.
        middle_pos = (self.size // 2, self.size // 2)
        self.highlight_circle = plt.Circle(
            middle_pos,
            radius=5,
            fill=True,
            linewidth=1,
            facecolor="#ff0000aa",
            edgecolor="#00000055"
        )
        self.highlight_circle_small = plt.Circle(middle_pos, radius=0.5, color="black")
        self.fft_ax.add_patch(self.highlight_circle)
        self.fft_ax.add_patch(self.highlight_circle_small)

        # The order of the sinusoids to draw.
        frequency_count = math.ceil(self.size / 2)
        self.frequencies_to_draw = [(x, y - frequency_count) for x in range(frequency_count) for y in range(self.size)]
        self.frequencies_to_draw.sort(key=SINUSOID_DRAW_ORDER)

    def animate(self, step):
        if step >= len(self.frequencies_to_draw):
            print("Finished")
            return

        print(step)

        x, y = self.frequencies_to_draw[step]

        # For real inputs (i.e. all images), the spectrum is conjugate symmetric.
        # When the complex sinusoid is added to its complex conjugate, the imaginary components equal out
        # and only the real component is left (with twice the amplitude).
        coefficient = self.fft[y, x] * (1 if x == 0 else 2)
        sinusoid = compute_2d_complex_sinusoid(self.size, x, y, coefficient)
        sinusoid = np.real(sinusoid)
        self.image += sinusoid

        # Display the images
        vmin, vmax = self.compute_value_range_for_brightness(x, y, coefficient)
        self.layer_imshow.set_data(sinusoid)
        self.layer_imshow.set_clim(vmin=vmin, vmax=vmax)
        self.image_imshow.set_data(self.image)

        self.layer_ax.set_title(f"Sinusoid freq x={x} y={y}")

        self.mark_fft_pixel_as_visited(x, y)
        self.mark_fft_pixel_as_visited(-x, y)
        self.fft_imshow.set_data(self.visited_fft_image)

        self.highlight_fft_pixel(x, y)

    def compute_value_range_for_brightness(self, x, y, coefficient):
        if x == y == 0:
            return 0, 255

        actual_brightness = np.abs(coefficient) / (self.size ** 2)
        target_rel_brightness = (actual_brightness / 255) ** (1 / BRIGHTNESS_BIAS) * BRIGHTNESS_FACTOR
        if target_rel_brightness > 1:
            target_rel_brightness = 1

        dynamic_range = max(actual_brightness, 2 * actual_brightness / target_rel_brightness)
        vmin = -actual_brightness
        vmax = dynamic_range - actual_brightness

        return vmin, vmax

    def get_fft_pixel_position(self, x, y):
        """ Returns the image pixel position corresponding to the component with the given x and y frequencies. """
        img_x = (x + self.size // 2) % self.size
        img_y = (y + self.size // 2) % self.size
        return img_x, img_y

    def mark_fft_pixel_as_visited(self, x, y):
        img_x, img_y = self.get_fft_pixel_position(x, y)
        pixel_color = VISITED_COLORMAP(self.normalized_fft_image[img_y, img_x])[:3]
        self.visited_fft_image[img_y, img_x] = pixel_color

    def highlight_fft_pixel(self, x, y):
        img_x, img_y = self.get_fft_pixel_position(x, y)
        self.highlight_circle.set_center((img_x, img_y))
        self.highlight_circle_small.set_center((img_x, img_y))


image = Image.open(INPUT_IMAGE_PATH)

# Convert to greyscale.
image = image.convert("L")
image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

image_data = np.array(image)
image_fft = np.fft.fft2(image_data)

# Layout
#   [ Original Image ] [ FFT Spectrum ]
#   [ Current Image  ] [ Sinusoid     ]

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0, 0].imshow(image_data, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Original Image")

animator = Animator(
    image_ax=ax[1, 0],
    layer_ax=ax[1, 1],
    fft_ax=ax[0, 1],
    fft=image_fft
)

current_step = 0
def draw_next_step():
    global current_step

    animator.animate(current_step)
    current_step += 1
    plt.draw()


def on_click(event):
    if event.button == 1:
        draw_next_step()


def on_key(event):
    if event.key == "right" or event.key == " ":
        draw_next_step()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

fig.suptitle("Click / Space / Right Arrow for next step")
fig.canvas.set_window_title("2D FFT Visualisation")

# animation = FuncAnimation(fig, animator.animate, init_func=lambda: None, frames=1000, interval=1000, repeat=False)

draw_next_step()

plt.show()
