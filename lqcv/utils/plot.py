import cv2

LQCV_PAUSE=False
def waitKey(delay=1):
    """A better waitKey that can pause video or image sequences."""
    global LQCV_PAUSE
    key = cv2.waitKey(0 if LQCV_PAUSE else delay)
    LQCV_PAUSE = True if key == ord(' ') else False
    return key

def cv2_imshow(im, delay=0, wname="p"):
    """A prepared cv2.imshow to reduce duplicate code."""
    cv2.imshow(wname, im)
    if waitKey(delay) == ord('q'):
        exit()


    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def plot_one_box(x, img, color=(0, 0, 255), label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] + t_size[1]),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
