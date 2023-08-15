import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import math


IMAGE_SIZE = 256


def compute_2d_complex_sinusoid(size, freq_x, freq_y, coefficient):
    values = np.arange(size)
    x, y = np.meshgrid(values, values)
    return coefficient * np.exp(2j * np.pi / size * (freq_x * x + freq_y * y)) / (size ** 2)


class Animation:
    def __init__(self, image_ax, layer_ax, size, fft):
        self.size = size
        self.image = np.zeros((size, size), dtype=np.complex)
        self.image_ax = image_ax
        self.layer_ax = layer_ax
        self.fft = fft

        # self.frequencies_to_draw = [(x-127, y-127) for x in range(size) for y in range(size)]

        frequency_count = math.ceil(size / 2)
        self.frequencies_to_draw = [(x, y-127) for x in range(frequency_count) for y in range(size)]
        self.frequencies_to_draw.sort(key=lambda pos: pos[0] ** 2 + pos[1] ** 2)

    def init(self):
        pass

    def animate(self, step):
        print(step)

        self.image_ax.clear()
        self.layer_ax.clear()

        x, y = self.frequencies_to_draw[step]

        sinusoid = compute_2d_complex_sinusoid(self.size, x, y, self.fft[y, x])
        # For real inputs (i.e. all images), the spectrum is conjugate symmetric.
        # When the complex sinusoid is added to its complex conjugate, the imaginary components equal out
        # and only the real component is left (with twice the amplitude).
        sinusoid = np.real(sinusoid) * (1 if x == 0 else 2)
        self.image += sinusoid

        self.layer_ax.imshow(np.real(sinusoid), cmap="gray")  # TODO
        self.image_ax.imshow(np.real(self.image).clip(0, 255), cmap="gray", vmin=0, vmax=255)  # TODO: Do I need to clip?

        self.layer_ax.set_title(f"Sinusoid freq x={x} y={y}")


image = Image.open("input/stop.jpg")

# Convert to greyscale
image = image.convert("L")
image = image.resize((256, 256))
# image.show()

image_data = np.array(image)
image_fft = np.fft.fft2(image_data)

# Layout
# [ Original Image ] [ FFT Spectrum ]
# [ Current Image  ] [ Sinusoid     ]

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0, 0].imshow(image_data, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Original Image")

ax[0, 1].imshow(np.log10(np.abs(np.fft.fftshift(image_fft)) + 1))
ax[0, 1].set_title("FFT")

animator = Animation(ax[1, 0], ax[1, 1], 256, image_fft)


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
