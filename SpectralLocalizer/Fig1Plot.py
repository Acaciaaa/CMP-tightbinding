from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

img1 = Image.open("/Users/ruiqi/Documents/tmp/currents/fig1a_sub1.png")
img2 = Image.open("/Users/ruiqi/Documents/tmp/currents/fig1a_sub2.png")

w1, h1 = img1.size
w2, h2 = img2.size

scale = w1 / w2 * 0.6
new_w2 = int(w2 * scale)
new_h2 = int(h2 * scale)
img2_resized = img2.resize((new_w2, new_h2))

offset_x = (w1 - new_w2) // 2

total_height = h1 + new_h2

def stitching():
    new_img = Image.new("RGB", (w1, total_height), color=(255, 255, 255))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2_resized, (offset_x, h1))

    new_img.save("/Users/ruiqi/Documents/tmp/currents/stitching.png")

def zoomin():
    img = mpimg.imread("/Users/ruiqi/Documents/tmp/currents/stitching.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    ax.axis("off")

    def map_point_top(point):
        return point

    def map_point_bottom(point):
        x_new = offset_x + point[0] * scale
        y_new = h1 + point[1] * scale
        return (x_new, y_new)

    p1 = (w1//2-96, h1//2 - 50)
    p2 = (w2//2-360, h2//2 - 195)
    p1 = map_point_top(p1)
    p2 = map_point_bottom(p2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=":", color="black", linewidth=1.5)

    p1 = (w1//2+88, h1//2 - 50)
    p2 = (w2//2+357, h2//2 - 195)
    p1 = map_point_top(p1)
    p2 = map_point_bottom(p2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=":", color="black", linewidth=1.5)

    #plt.show()
    plt.savefig("/Users/ruiqi/Documents/tmp/currents/zoomin.png", bbox_inches="tight", pad_inches=0, dpi=300)

def uparrow():
    img = mpimg.imread("/Users/ruiqi/Documents/tmp/currents/zoomin.png")
    fig, ax = plt.subplots()
    ax.imshow(img)

    start = (728, 380)   # 弯曲箭头的起点（文字下面）
    end = (718, 285)     # 弯曲箭头的终点（箭头尖）

    # 创建弯曲箭头
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        connectionstyle="arc3,rad=0.4",  # 控制弯曲程度和方向
        arrowstyle='-|>',
        color='black',
        lw=1,
        mutation_scale=10
    )
    ax.add_patch(arrow)

    ax.text(710, 230, r"$I_\text{circ}$",
            fontsize=12, color='black', ha='center', va='top')

    plt.axis('off')  # 如果你不想要坐标轴
    #plt.show()
    plt.savefig("/Users/ruiqi/Documents/tmp/currents/fig1a.png", bbox_inches="tight", pad_inches=0, dpi=300)

#stitching()
#zoomin()
uparrow()