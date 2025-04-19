from PIL import Image

img1 = Image.open("/Users/ruiqi/Documents/tmp/currents/fig1a_sub1.png")
img2 = Image.open("/Users/ruiqi/Documents/tmp/currents/fig1a_sub2.png")

w1, h1 = img1.size
w2, h2 = img2.size

scale = h1 / h2 * 0.6
new_w2 = int(w2 * scale)
new_h2 = int(h2 * scale)
img2_resized = img2.resize((new_w2, new_h2))

offset_y = (h1 - new_h2) // 2

total_width = w1 + new_w2
new_img = Image.new("RGB", (total_width, h1), color=(255, 255, 255))

new_img.paste(img1, (0, 0))
new_img.paste(img2_resized, (w1, offset_y))

new_img.save("/Users/ruiqi/Documents/tmp/currents/tmp.png")
def map_point_left(point):
    return point

def map_point_right(point):
    x_new = w1 + point[0] * scale
    y_new = offset_y + point[1] * scale
    return (x_new, y_new)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt

img = mpimg.imread("/Users/ruiqi/Documents/tmp/currents/tmp.png")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img)
ax.axis("off")

p1 = (w1//2, h1//2+105)
p2 = (w2//2, h2//2+410)
p1 = map_point_left(p1)
p2 = map_point_right(p2)
ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linestyle=":",
        color="black",
        linewidth=1.5,
)
p1 = (w1//2, h1//2-105)
p2 = (w2//2, h2//2-410)
p1 = map_point_left(p1)
p2 = map_point_right(p2)
ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linestyle=":",
        color="black",
        linewidth=1.5,
)

plt.savefig("/Users/ruiqi/Documents/tmp/currents/fig1a.png", bbox_inches="tight", pad_inches=0, dpi=300)