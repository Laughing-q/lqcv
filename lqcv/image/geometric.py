import cv2

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imrescale(
    img,
    new_wh,
    keep_ratio=True,
    return_scale=False,
    interpolation="bilinear",
):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        new_wh (tuple[int]): Target size (w, h).
        keep_ratio (bool): Whether to keep the aspect ratio.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    nw, nh = new_wh
    if keep_ratio:
        scale_factor = min(nw / w, nh / h)
        im = cv2.resize(
            im,
            (int(w * scale_factor), int(h * scale_factor)),
            interpolation=cv2_interp_codes[interpolation],
        )
    else:
        im = cv2.resize(im, (nw, nh), interpolation=cv2_interp_codes[interpolation])

    if return_scale:
        nh, nw = im.shape[:2]
        w_scale = nw / w
        h_scale = nh / h
        return im, w_scale, h_scale
    return im
