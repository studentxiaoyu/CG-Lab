import taichi as ti

# 1. 统一初始化 Taichi (必须在最前面执行)
ti.init(arch=ti.gpu)

# 2. 升级为 16:9 高清宽屏比例 (完美适配全屏/最大化，防止物体变形)
res_x, res_y = 1280, 720

# 3. 渲染结果的像素显存缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 4. 全局材质参数 (GUI 控制)
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

# 5. 选做模块的高级特效开关 (GUI 控制)
use_blinn_phong = ti.field(ti.i32, shape=())
use_shadow = ti.field(ti.i32, shape=())

def init_default_parameters():
    """初始化 UI 参数的默认值"""
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0
    use_blinn_phong[None] = 1
    use_shadow[None] = 1