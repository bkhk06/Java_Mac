import matplotlib.pyplot as plt
# 创建 figure
fig, ax = plt.subplots()
fig = plt.figure(figsize=(6, 8))

# 成员函数
fig.set_tight_layout(True)
fig.get_dpi()
fig.get_size_inches()

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

f = plt.figure(figsize=(8, 8))

# 接受一个时间参数，
def make_frame_mpl(t):
    ...
    return mplfig_to_npimage(f)
                # 所有的图像都在同一个figure上

animation = mpy.VideoClip(make_frame_mpl, duration=5)
animation.write_gif("animation-94a2c1ff.gif", fps=20)
        # 这样其实就可以做一个动态图出来了；
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn

fig, ax = plt.subplots()

x = np.arange(0, 20, .01)
ax.scatter(x, x+np.random.normal(0, 3, len(x)))
line, = ax.plot(x, x-5, 'r-', lw=2)


# i => ith frame
def update(i):
    line.set_ydata(x-5+i)
    ax.set_xlabel('frame {0}'.format(i))
    return line, ax

if __name__ == '__main__':

    animation = FuncAnimation(fig, update, frames=np.arange(0, 20), interval=200)
        # 200ms 的间隔，相当于 5fps，一秒 5 帧
    plt.show()
