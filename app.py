import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import math

IMAGE_SIZE = 256

# Constant with which the sinusoid brightness is multiplied
BRIGHTNESS_FACTOR = 2

# A non-linear parameter that skews the brightness curve for sinusoids.
# For values greater than 1 darker sinusoids appear brighter.
BRIGHTNESS_BIAS = 2

COLORMAP = plt.get_cmap("viridis")
VISITED_COLORMAP = plt.get_cmap("plasma")


def compute_2d_complex_sinusoid(size, freq_x, freq_y, coefficient):
    values = np.arange(size)
    x, y = np.meshgrid(values, values)
    return coefficient * np.exp(2j * np.pi / size * (freq_x * x + freq_y * y)) / (size ** 2)


class Animator:
    def __init__(self, image_ax, layer_ax, fft_ax, fft):
        self.size = len(fft)
        self.image_ax = image_ax
        self.layer_ax = layer_ax
        self.fft_ax = fft_ax
        self.fft = fft

        self.image = np.zeros((self.size, self.size), dtype=np.float)
        # self.visited_fft_image = np.log10(np.abs(np.fft.fftshift(self.fft)) + 1)

        self.normalized_fft_brightness = np.log10(np.abs(self.fft) + 1)
        self.normalized_fft_brightness /= np.max(self.normalized_fft_brightness)

        self.visited_fft_image = np.fft.fftshift(self.normalized_fft_brightness)
        self.visited_fft_image /= np.max(self.visited_fft_image)
        self.visited_fft_image = COLORMAP(self.visited_fft_image)[:, :, :3]

        self.fft_ax.set_title("FFT")
        self.fft_ax.imshow(self.visited_fft_image)

        frequency_count = math.ceil(self.size / 2)
        self.frequencies_to_draw = [(x, y - 127) for x in range(frequency_count) for y in range(self.size)]
        self.frequencies_to_draw.sort(key=lambda pos: pos[0] ** 2 + pos[1] ** 2)

    def init(self):
        pass

    def animate(self, step):
        print(step)

        self.image_ax.clear()
        self.layer_ax.clear()

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
        self.layer_ax.imshow(np.real(sinusoid), cmap="gray", vmin=vmin, vmax=vmax)
        self.image_ax.imshow(np.real(self.image), cmap="gray", vmin=0, vmax=255)

        self.layer_ax.set_title(f"Sinusoid freq x={x} y={y}")

        self.highlight_current_area(x, y)
        self.highlight_current_area(-x, y)
        self.fft_ax.imshow(self.visited_fft_image)

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

    def highlight_current_area(self, x, y):
        pixel_color = VISITED_COLORMAP(self.normalized_fft_brightness[y, x])[:3]
        img_x = (x + self.size // 2) % self.size
        img_y = (y + self.size // 2) % self.size
        self.visited_fft_image[img_x, img_y] = pixel_color



image = Image.open("input/stop.jpg")

# Convert to greyscale
image = image.convert("L")
image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
# image.show()

image_data = np.array(image)
image_fft = np.fft.fft2(image_data)

# Layout
# [ Original Image ] [ FFT Spectrum ]
# [ Current Image  ] [ Sinusoid     ]

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
fig.suptitle("Click / Space / Right for next step")

# animation = FuncAnimation(fig, animator.animate, init_func=animator.init, frames=1000, interval=1000, repeat=False)

draw_next_step()

plt.show()
