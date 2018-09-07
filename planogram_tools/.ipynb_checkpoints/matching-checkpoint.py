import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import openpyxl
from pathlib import Path


def match_planogram(
    img1,
    img2,
    min_matches=10,
    method="orb",
    matcher="flann",
    plot=True,
    plot_matches=False,
):
    """Match a planogram image to a photo, and return an undistorted extract of the display from the photo"""

    img1_shape_orig = img1.shape

    # Resize img1 so it fits inside the img2
    min_idx = np.argmax(img1.shape[:2])
    r = min(1, img2.shape[min_idx] / img1.shape[min_idx])
    img1 = cv.resize(
        img1.copy(), tuple((np.array(img1.shape[:2][::-1]) * r).astype(int))
    )

    detector = {
        "sift": cv.xfeatures2d.SIFT_create(),
        "surf": cv.xfeatures2d.SURF_create(),
        "orb": cv.ORB_create(
            nfeatures=1_000_000, scaleFactor=1.1
        ),  # , scoreType=cv.NORM_HAMMING2),
    }[method]

    # Find the keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if method == "orb":
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=6)

    search_params = dict(checks=100)

    # Initiate matcher
    if matcher == "brute":
        matcher = cv.BFMatcher()
    else:
        matcher = cv.FlannBasedMatcher(index_params, search_params)

    # Match features
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    # store all the good matches as per Lowe's ratio test.
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    # print('Found', len(good), 'good matches')

    if len(good) > 0:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(
            src_pts, dst_pts, cv.RANSAC, 5.0
        )  # RANSAC, LMEDS, RHO
        matchesMask = mask.ravel().tolist()

        # Get list of source and destination corner points for the transformation
        h, w, _ = img1.shape
        output_rect = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)
        input_rect = cv.perspectiveTransform(output_rect, M)

        h, w, _ = img1_shape_orig
        output_rect_scaled = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)

        # Compute the transformation matrix
        transformation = cv.getPerspectiveTransform(input_rect, output_rect_scaled)

        # Transform the image to un-skew it
        warp = cv.warpPerspective(img2, transformation, img1_shape_orig[:2][::-1])

        # Plot the un-skewed image if required
        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(warp)
            plt.show()

        # Plot matches between the two images if required
        if plot_matches:
            img2_border = cv.polylines(
                img2.copy(), [np.int32(input_rect)], True, 255, 10, cv.LINE_AA
            )
            fig, ax = plt.subplots(figsize=(18, 18))
            img3 = cv.drawMatches(
                img1,
                kp1,
                img2_border,
                kp2,
                good,
                None,
                # matchColor=(0, 255, 0),  # draw matches in green color
                # singlePointColor=(255,0,0),
                matchesMask=matchesMask,  # draw only inliers
                flags=2,
            )
            plt.imshow(img3, "gray")
            plt.show()

        return warp, transformation, len(good)


"""

inv = cv.invert(transformation)[1]

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(image)


rects = []

for slot in planogram["slots"]:
    l = slot["left"]
    t = slot["top"]
    w = slot["width"]
    h = slot["height"]
    r = slot["left"] + slot["width"]
    b = slot["top"] + slot["height"]

    points = (l, t), (r, t), (l, b), (r, b)

    def invtrans(x, y):
        n_x, n_y, s = inv @ (x, y, 1)
        return n_x / s, n_y / s

    new_points = [invtrans(*point) for point in points]

    xs = [x for x, y in new_points]
    ys = [y for x, y in new_points]

    rects.append(
        Rectangle(
            (min(xs), min(ys)),
            max(xs) - min(xs),
            max(ys) - min(ys),
            fc="none",
            linewidth=1,
            ec="r",
        )
    )


rect_collect = PatchCollection(rects, match_original=True)
ax.add_collection(rect_collect)
plt.show()
"""
