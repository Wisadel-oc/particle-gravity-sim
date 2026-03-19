import taichi as ti
import numpy as np
from .config import (
    SCREEN_SIZE, BACKGROUND_COLOR, PARTICLE_RADIUS,
    MOUSE_GRAVITY_STRENGTH, N_PARTICLES
)
from .physics import (
    pos, vel, update_physics_base,
    apply_mouse_force, apply_particle_interactions
)

def main():
    gui = ti.GUI("粒子引力仿真", res=SCREEN_SIZE)
    print("操作说明：")
    print("- 按住空格键：粒子变红 + 向鼠标聚拢")
    print("- 按【ESC】退出")
    print("- 仿真规模为400粒子数")

    color_np = np.ones((N_PARTICLES, 3), dtype=np.float32) * [0.88, 0.88, 1.0]
    mouse_active = gui.is_pressed(ti.GUI.SPACE)

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False

    while gui.running:
        events = gui.get_events(ti.GUI.PRESS)
        release_events = gui.get_events(ti.GUI.RELEASE)
        
        # 处理按下事件
        for e in events:
            if e.key == ti.GUI.SPACE:
                mouse_active = True
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 处理释放事件
        for e in release_events:
            if e.key == ti.GUI.SPACE:
                mouse_active = False #疑似未生效

        # 鼠标坐标
        mouse_x, mouse_y = gui.get_cursor_pos()

        # 交互逻辑
        if mouse_active:
            color_np[:] = [1.0, 0.6, 0.6]
            apply_mouse_force(mouse_x, mouse_y, MOUSE_GRAVITY_STRENGTH)
        else:
            color_np[:] = [0.88, 0.88, 1.0]

        # 粒子间作用 + 物理更新
        apply_particle_interactions()
        for _ in range(5):
            update_physics_base()

        # 数据同步
        pos_np = pos.to_numpy()

        # 渲染
        gui.clear(BACKGROUND_COLOR)
        color_hex = (color_np * 255).astype(np.uint32)
        color_hex = (color_hex[:, 0] << 16) + (color_hex[:, 1] << 8) + color_hex[:, 2]
        gui.circles(pos=pos_np, radius=PARTICLE_RADIUS, color=color_hex)

        gui.show()

if __name__ == "__main__":
    main()