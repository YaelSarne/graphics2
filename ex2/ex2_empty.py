import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
import random

from utils import *
import numpy as np
import random


def harris_corner_detector(im):
    """
    Implements the harris corner detection algorithm.
    :param im: A 2D array representing a grayscale image.
    :return: An array with shape (N,2), where its ith entry is the [x,y] coordinates of the ith corner point.
    """
    im = im.astype(np.float64)
    vec_x = np.array([[1, 0, -1]], dtype=np.float64)
    vec_y = np.array([[1], [0], [-1]], dtype=np.float64)

    Ix = convolve2d(im, vec_x, mode='same', boundary='symm')
    Iy = convolve2d(im, vec_y, mode='same', boundary='symm')

    IxIy = np.multiply(Ix, Iy)
    IyIx = np.multiply(Iy, Ix)
    IyIy = np.multiply(Iy, Iy)
    IxIx = np.multiply(Ix, Ix)
    
    IxIy_blured = blur_spatial(IxIy, 3)
    IyIx_blured = blur_spatial(IyIx, 3)
    IyIy_blured = blur_spatial(IyIy, 3)
    IxIx_blured = blur_spatial(IxIx, 3)

    det_M = IxIx_blured * IyIy_blured - IxIy_blured * IyIx_blured
    trace_M = IxIx_blured + IyIy_blured

    alpha = 0.04
    R_image = det_M - alpha * (trace_M ** 2)
    R_image[R_image < 0] = 0

    bool_image_max = non_maximum_suppression(R_image)
    ys, xs = np.nonzero(bool_image_max)
    points = np.stack([xs, ys], axis=1)
    return points


def feature_descriptor(im, points, desc_rad=3):
    """
    Samples descriptors at the given feature points.
    :param im: A 2D array representing a grayscale image.
    :param points: An array with shape (N,2) representing feature points coordinates in the image.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: An array of 2D patches, each patch i representing the descriptor of point i.
    """
    patches = []
    axis = np.arange(-desc_rad, desc_rad + 1)
    dx, dy = np.meshgrid(axis, axis)
    for point in points:
        x_cords = dx + point[0]
        y_cords = dy + point[1]
        curr_patch = map_coordinates(im, [y_cords.ravel(), x_cords.ravel()])
        mean = np.mean(curr_patch)
        curr_patch = curr_patch - mean
        norm = np.linalg.norm(curr_patch)
        if norm:
            curr_patch = curr_patch / norm
        curr_patch = np.reshape(curr_patch, (7, 7))
        patches.append(curr_patch)
    return patches


def find_features(im):
    """
    Detects and extracts feature points from a specific pyramid level.
    :param im: A 2D array representing a grayscale image.
    :return: A list containing:
             1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                These coordinates are provided at the original image level.
            2) A feature descriptor array with shape (N,K,K)
    """
    pyramid = build_gaussian_pyramid(im, 3, 3)
    corners = spread_out_corners(im, 7, 7, 12, harris_corner_detector)
    corners_level3 = corners / 4
    descriptions = feature_descriptor(pyramid[2], corners_level3, 3)
    return [corners, descriptions]

def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    d1_flat = np.array(desc1.reshape(desc1.shape[0], -1))
    d2_flat = desc2.reshape(desc2.shape[0], -1)
    S = np.dot(d1_flat, d2_flat.T)
    top2_for1 = np.argpartition(S, -2, axis=1)[:, -2:]
    top2_for2 = np.argpartition(S, -2, axis=0)[-2:, :]
    desc1_mask = np.zeros_like(S, dtype=bool)
    desc1_mask[np.arange(len(desc1))[:, None], top2_for1] = True
    desc2_mask = np.zeros_like(S, dtype=bool)
    desc2_mask[np.arange(len(desc2))[:, None], top2_for2] = True
    score_mask = S > min_score
    final_mask = desc1_mask & desc2_mask & score_mask
    idx1, idx2 = np.where(final_mask)
    return [idx1, idx2]

def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pass

def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    N = points1.shape[0]
    best_inliers = np.array([], dtype=int)
    best_H = None

    for iter in num_iter:
        i, j = random.sample(range(N), 2)

        P1 = points1[[i, j], :]
        P2 = points2[[i, j], :]

        H12 = estimate_rigid_transform(P1, P2, translation_only=translation_only)
        p1_transformed = apply_homography(points1, H12)

        d2 = np.sum((p1_transformed - points2) ** 2, axis=1)
        inliers = np.where(d2 < inlier_tol)[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_H = H12

    if best_H is None:
        best_H = np.eye(3)

    if best_inliers.size >= 2:
        best_H = estimate_rigid_transform(points1[best_inliers], points2[best_inliers])
        
    if best_H[2,2] !=0:
        best_H = best_H / best_H[2,2]
    
    return [best_H, best_inliers]

def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    combined_im = np.hstack([im1, im2])
    plt.imshow(combined_im, cmap='gray')
    width_im1 = im1.shape[1]
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0] + width_im1, points2[:, 1]
    for i in range(len(points1)):
        color = 'b' if i in inliers else 'y'
        plt.plot([x1[i], x2[i]], [y1[i], y2[i]], color=color, lw=0.6, alpha=0.5)
    plt.scatter(x1, y1, c='r', s=5)
    plt.scatter(x2, y2, c='r', s=5)
    plt.axis('off') 
    plt.show()


def accumulate_homographies(H_successive, m):
    """
    Convert a list of successive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    pass


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0, 0],[w-1, 0],[0, h-1],[w-1, h-1]], dtype=float)
    transformed = apply_homography(corners, homography)

    min_xy = np.min(transformed, axis=0)  #[min_x, min_y]
    max_xy = np.max(transformed, axis=0)  #[max_x, max_y]

    return np.vstack([min_xy, max_xy])

def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h, w = image.shape
    (min_x, min_y), (max_x, max_y) = compute_bounding_box(homography, w, h)
    x = np.arange(np.floor(min_x), np.ceil(max_x)+1)
    y = np.arange(np.floor(min_y), np.ceil(max_y)+1)
    X, Y = np.meshgrid(x,y)

    H_inv = np.linalg.inv(homography)
    flattened = np.stack([X.ravel(), Y.ravel()], axis=1)
    src_points = apply_homography(flattened, H_inv)
    
    Xsrc = src_points[:, 0].reshape(X.shape)
    Ysrc = src_points[:, 1].reshape(Y.shape)
    vals = map_coordinates(image, [Ysrc.ravel(), Xsrc.ravel()],
        order=1, mode='constant',cval=0.0)
    warped = vals.reshape(X.shape)
    return warped

def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    warped_channels = []
    for c in range(3):
        warped_c = warp_channel(image[:, :, c], homography)
        warped_channels.append(warped_c)
    warped_image = np.stack(warped_channels, axis=2)
    return warped_image


##################################################################################################


def align_images(files, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in files:
        image = read_image(file, 1)
        points_and_descriptors.append(find_features(image))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
        points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
        desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = match_features(desc1, desc2, .7)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

        Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    homographies = np.stack(accumulated_homographies)
    frames_for_panoramas = filter_homographies_with_translation(homographies, minimum_right_translation=5)
    homographies = homographies[frames_for_panoramas]
    return frames_for_panoramas, homographies

def generate_panoramic_images(data_dir, file_prefix, num_images, out_dir, number_of_panoramas, translation_only=False):
    """
    combine slices from input images to panoramas.
    The naming convention for a sequence of images is file_prefixN.jpg, where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    :param out_dir: path to output panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """

    file_prefix = file_prefix
    files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    files = list(filter(os.path.exists, files))
    print('found %d images' % len(files))
    image = read_image(files[0], 1)
    h, w = image.shape

    frames_for_panoramas, homographies = align_images(files, translation_only)

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    bounding_boxes = np.zeros((frames_for_panoramas.size, 2, 2))
    for i in range(frames_for_panoramas.size):
        bounding_boxes[i] = compute_bounding_box(homographies[i], w, h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(bounding_boxes, axis=(0, 1))
    bounding_boxes -= global_offset

    slice_centers = np.linspace(0, w, number_of_panoramas + 2, endpoint=True, dtype=np.int32)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
        slice_center_2d = np.array([slice_centers[i], h // 2])[None, :]
        # homography warps the slice center to the coordinate system of the middle image
        warped_centers = [apply_homography(slice_center_2d, h) for h in homographies]
        # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
        warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(bounding_boxes, axis=(0, 1)).astype(np.int32) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int32)

    panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(frames_for_panoramas):
        # warp every input image once, and populate all panoramas
        image = read_image(files[frame_index], 2)
        warped_image = warp_image(image, homographies[i])
        x_offset, y_offset = bounding_boxes[i][0].astype(np.int32)
        y_bottom = y_offset + warped_image.shape[0]

        for panorama_index in range(number_of_panoramas):
            # take strip of warped image and paste to current panorama
            boundaries = x_strip_boundary[panorama_index, i:i + 2]
            image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
            x_end = boundaries[0] + image_strip.shape[1]
            panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    os.makedirs(out_dir, exist_ok=True)
    for i, panorama in enumerate(panoramas):
        plt.imsave('%s/panorama%02d.png' % (out_dir, i + 1), panorama)


if __name__ == "__main__":
    import ffmpeg
    video_name = "mt_cook.mp4"
    video_name_base = video_name.split('.')[0]
    os.makedirs(f"dump/{video_name_base}", exist_ok=True)
    ffmpeg.input(f"videos/{video_name}").output(f"dump/{video_name_base}/{video_name_base}%03d.jpg").run()
    num_images = len(os.listdir(f"dump/{video_name_base}"))
    print(f"Generated {num_images} images")

    # Visualize feature points on two sample images
    print("Extracting and visualizing feature points...")
    image1 = read_image(f"dump/{video_name_base}/{video_name_base}200.jpg", 1)
    image2 = read_image(f"dump/{video_name_base}/{video_name_base}300.jpg", 1)

    # Extract feature points and descriptors
    points1, desc1 = find_features(image1)
    points2, desc2 = find_features(image2)

    # Visualize points on first image
    print(f"Found {len(points1)} feature points in image 1")
    visualize_points(image1, points1)

    # Visualize points on second image
    print(f"Found {len(points2)} feature points in image 2")
    visualize_points(image2, points2)

    # Match features between the two images
    print("Matching features between images...")
    ind1, ind2 = match_features(desc1, desc2, 0.6)
    matched_points1 = points1[ind1]
    matched_points2 = points2[ind2]
    print(f"Found {len(ind1)} matches")

    # Run RANSAC to find inliers
    H12, inliers = ransac_homography(matched_points1, matched_points2, 100, 6, translation_only=False)
    print(f"Found {len(inliers)} inliers out of {len(matched_points1)} matches")

    # Display matches with inliers and outliers
    display_matches(image1, image2, matched_points1, matched_points2, inliers)

    # Generate panoramic images
    #print("\nGenerating panoramic images...")
    #generate_panoramic_images(f"dump/{video_name_base}/", video_name_base,
    #                          num_images=num_images, out_dir=f"out/{video_name_base}", number_of_panoramas=3)