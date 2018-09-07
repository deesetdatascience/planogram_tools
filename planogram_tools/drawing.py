import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import openpyxl
from pathlib import Path


def feather(image):
    # Feather the corners of an image
    astro = image
    l_row, l_col, _ = astro.shape
    nb_channel = 3
    rows, cols = np.mgrid[:l_row, :l_col]
    radius = np.sqrt((rows - l_row / 2) ** 2 + (cols - l_col / 2) ** 2)
    alpha_channel = np.zeros((l_row, l_col))
    r_min, r_max = 0.9 * radius.max(), 1 * radius.max()
    alpha_channel[radius < r_min] = 1
    alpha_channel[radius > r_max] = 0
    gradient_zone = np.logical_and(radius >= r_min, radius <= r_max)
    alpha_channel[gradient_zone] = (r_max - radius[gradient_zone]) / (r_max - r_min)
    alpha_channel *= 255
    feathered = np.empty((l_row, l_col, nb_channel + 1), dtype=np.uint8)
    feathered[..., :3] = astro[..., :3]
    feathered[..., -1] = alpha_channel[:]
    return feathered


def draw_planogram(plan, image_out_path=None, show_missing=False, show=True, dpi=200):
    """Draw a planogram from a python planogram object"""

    # Set up the figure
    fig, ax = plt.subplots(
        figsize=(plan["width"] / dpi, plan["height"] / dpi), facecolor=plan["bg_colour"]
    )
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # Set the background colour
    ax.set_facecolor(plan["bg_colour"])

    # Hide the axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Draw each product in its slot
    for slot in plan["slots"]:

        x1 = slot["left"]
        y1 = slot["top"]
        w = slot["width"]
        h = slot["height"]

        if "image_path" in slot.keys():
            image_path = slot["image_path"]
            # image = feather( imread(image_path) )
            image = cv.imread(str(image_path))[:, :, ::-1]
            ax.imshow(image, extent=(x1, x1 + w, y1, y1 + h), origin="lower")
        else:
            if show_missing:
                ax.add_patch(
                    Rectangle((x1, y1), w, h, ec="#aaaaaa", linestyle=":", fc="none")
                )
            pass

    # Set the image width and height
    ax.set_xlim(0, plan["width"] + 1)
    ax.set_ylim(plan["height"] + 1, 0)
    # Hide the figure frame
    ax.set_frame_on(False)

    # Optionally show the plot
    if show:
        plt.show()

    # Optionally save the plot
    if image_out_path:
        fig.savefig(
            image_out_path,
            dpi=dpi,
            pad_inches=0,
            # facecolor='#000000',
            facecolor=fig.get_facecolor(),
            # transparent=True,
        )

    plt.close()
