import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# ===================== 物理与网格参数 =====================

N = 20
mass = 1.0
dt = 5e-4

k_s = 10000.0
k_d = 1.0

gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 球体碰撞参数
sphere_radius = 0.22
sphere_restitution = 0.2
sphere_friction = 0.98

# 弹簧数量上限：
# 结构弹簧 + 剪切弹簧 + 弯曲弹簧，N*N*8 足够覆盖 20x20 的情况
max_springs = N * N * 8

# ===================== Taichi 数据场 =====================

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

# 隐式欧拉缓存
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

# 碰撞球体，用 field 是为了能被 GGUI 渲染
sphere_center = ti.Vector.field(3, dtype=float, shape=1)

# ===================== 初始化 =====================

@ti.kernel
def init_positions():
    """初始化质点位置、速度、受力和固定点"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        x[idx] = ti.Vector([
            i * 0.05 - 0.5,
            0.8,
            j * 0.05 - 0.5
        ])

        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])

        # 固定布料上边两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.kernel
def init_sphere():
    """初始化球体位置"""
    sphere_center[0] = ti.Vector([0.0, 0.28, 0.0])


@ti.func
def add_spring(idx_a: int, idx_b: int, spring_type: int):
    """添加一根弹簧"""
    c = ti.atomic_add(num_springs[None], 1)

    spring_pairs[c] = ti.Vector([idx_a, idx_b])
    spring_lengths[c] = (x[idx_a] - x[idx_b]).norm()
    spring_types[c] = spring_type


@ti.kernel
def init_springs(use_shear: ti.i32, use_bending: ti.i32):
    """
    初始化弹簧拓扑。

    spring_type:
    0 = Structural 结构弹簧
    1 = Shear 剪切弹簧
    2 = Bending 弯曲弹簧
    """
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # ---------------- 结构弹簧 Structural ----------------
        # 横向相邻
        if i < N - 1:
            idx_right = (i + 1) * N + j
            add_spring(idx, idx_right, 0)

        # 纵向相邻
        if j < N - 1:
            idx_down = i * N + (j + 1)
            add_spring(idx, idx_down, 0)

        # ---------------- 剪切弹簧 Shear ----------------
        # 对角线弹簧，用于抵抗布料斜向变形
        if use_shear == 1:
            if i < N - 1 and j < N - 1:
                idx_diag_1 = (i + 1) * N + (j + 1)
                add_spring(idx, idx_diag_1, 1)

            if i < N - 1 and j > 0:
                idx_diag_2 = (i + 1) * N + (j - 1)
                add_spring(idx, idx_diag_2, 1)

        # ---------------- 弯曲弹簧 Bending ----------------
        # 间隔一个质点的弹簧，用于抵抗布料过度弯折
        if use_bending == 1:
            if i < N - 2:
                idx_far_right = (i + 2) * N + j
                add_spring(idx, idx_far_right, 2)

            if j < N - 2:
                idx_far_down = i * N + (j + 2)
                add_spring(idx, idx_far_down, 2)


@ti.kernel
def init_spring_indices():
    """初始化渲染用线段索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]


def init_cloth(use_shear=True, use_bending=True):
    """Python 层顺序调用初始化 kernel，保证 GPU 状态同步"""
    num_springs[None] = 0

    init_positions()
    init_sphere()
    init_springs(int(use_shear), int(use_bending))
    init_spring_indices()


# ===================== 力学计算 =====================

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """
    计算重力、阻尼力和弹簧力。

    注意：
    弹簧力会同时作用在两个端点上，因此需要 atomic_add，
    否则多个线程可能同时写入同一个质点的 force。
    """
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]

        pos_a = pos[idx_a]
        pos_b = pos[idx_b]

        d = pos_a - pos_b
        dist = d.norm()

        if dist > 1e-6:
            direction = d / dist
            rest_len = spring_lengths[i]

            f_spring = -k_s * (dist - rest_len) * direction

            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)


@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()

    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity


@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """
    球体碰撞处理。

    如果质点进入球体内部：
    1. 把质点推出到球面上；
    2. 去除朝向球体内部的速度分量；
    3. 加入少量摩擦和反弹。
    """
    offset = pos[idx] - sphere_center[0]
    dist = offset.norm()

    if dist < sphere_radius:
        normal = ti.Vector([0.0, 1.0, 0.0])

        if dist > 1e-6:
            normal = offset / dist

        # 位置投影到球面
        pos[idx] = sphere_center[0] + normal * sphere_radius

        # 处理速度
        vn = vel[idx].dot(normal)

        if vn < 0.0:
            # 去掉向球内部的速度，并保留少量反弹
            vel[idx] -= (1.0 + sphere_restitution) * vn * normal

        # 简单摩擦，削弱贴着球面的滑动速度
        vel[idx] *= sphere_friction


# ===================== 积分求解器 =====================

@ti.kernel
def step_explicit(collision_enabled: ti.i32):
    """显式欧拉"""
    compute_forces_on(x, v, f)

    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt

            clamp_velocity(v, i)

            if collision_enabled == 1:
                resolve_sphere_collision(x, v, i)


@ti.kernel
def step_semi_implicit(collision_enabled: ti.i32):
    """半隐式欧拉"""
    compute_forces_on(x, v, f)

    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

            x[i] += v[i] * dt

            if collision_enabled == 1:
                resolve_sphere_collision(x, v, i)


@ti.kernel
def step_implicit_iter(collision_enabled: ti.i32):
    """
    隐式欧拉近似版本。

    使用定点迭代：
    先预测 x_next 和 v_next，
    然后用未来状态重新计算力，
    迭代若干次后写回。
    """
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]

    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)

        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)

                x_next[i] = x[i] + v_next[i] * dt

                if collision_enabled == 1:
                    resolve_sphere_collision(x_next, v_next, i)

    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] = v_next[i]
            x[i] = x_next[i]


# ===================== 主函数 =====================

def main():
    use_shear = True
    use_bending = True
    use_collision = True

    init_cloth(use_shear, use_bending)

    window = ti.ui.Window("Mass Spring System - Bonus Version", (900, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.2, 0.0)

    current_method = 1
    paused = False

    while window.running:
        # ================= GUI 控制面板 =================

        window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.48)

        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = 0
            init_cloth(use_shear, use_bending)

        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = 1
            init_cloth(use_shear, use_bending)

        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = 2
            init_cloth(use_shear, use_bending)

        window.GUI.text("")

        if window.GUI.button("Shear Spring: " + ("ON" if use_shear else "OFF")):
            use_shear = not use_shear
            init_cloth(use_shear, use_bending)

        if window.GUI.button("Bending Spring: " + ("ON" if use_bending else "OFF")):
            use_bending = not use_bending
            init_cloth(use_shear, use_bending)

        if window.GUI.button("Sphere Collision: " + ("ON" if use_collision else "OFF")):
            use_collision = not use_collision
            init_cloth(use_shear, use_bending)

        window.GUI.text("")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth(use_shear, use_bending)

        window.GUI.text("")
        window.GUI.text("Bonus Features:")
        window.GUI.text("- Shear springs")
        window.GUI.text("- Bending springs")
        window.GUI.text("- Sphere collision")

        window.GUI.end()

        # ================= 物理更新 =================

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit(int(use_collision))
                elif current_method == 1:
                    step_semi_implicit(int(use_collision))
                elif current_method == 2:
                    step_implicit_iter(int(use_collision))

        # ================= 渲染 =================

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

        # 布料质点
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))

        # 弹簧线框
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        # 碰撞球体
        if use_collision:
            scene.particles(sphere_center, radius=sphere_radius, color=(1.0, 0.4, 0.2))

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()