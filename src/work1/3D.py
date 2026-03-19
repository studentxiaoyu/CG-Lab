import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# 声明 Taichi 的 Field 来存储顶点和转换后的屏幕坐标
# 把shape=3变成8
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    模型变换矩阵：绕 Y 轴旋转 (展现真正的 3D 效果)
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    # 绕 Y 轴旋转的矩阵公式
    return ti.Matrix([
        [ c,  0.0,  s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换矩阵：将相机移动到原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    透视投影矩阵
    """
    # 视线看向 -Z 轴，实际坐标为负
    n = -zNear
    f = -zFar
    
    # 视角转化为弧度并求出 t, b, r, l
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 1. 挤压矩阵: 透视平截头体 -> 长方体
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 2. 正交投影矩阵: 缩放与平移至 [-1, 1]^3
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    
    # 返回组合矩阵
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    """
    在并行架构上计算顶点的坐标变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP 矩阵：右乘原则
    mvp = proj @ view @ model
    
    #循环的3改成8
    for i in range(8):
        v = vertices[i]
        # 补全齐次坐标
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法，转化为 NDC 坐标 [-1, 1]
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：映射到 GUI 的 [0, 1] x [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化三角形顶点（改成八个）
    # 前面 4 个顶点 (Z = 1.0)
    vertices[0] = [-1.0, -1.0,  1.0]
    vertices[1] = [ 1.0, -1.0,  1.0]
    vertices[2] = [ 1.0,  1.0,  1.0]
    vertices[3] = [-1.0,  1.0,  1.0]
    # 后面 4 个顶点 (Z = -1.0)
    vertices[4] = [-1.0, -1.0, -1.0]
    vertices[5] = [ 1.0, -1.0, -1.0]
    vertices[6] = [ 1.0,  1.0, -1.0]
    vertices[7] = [-1.0,  1.0, -1.0]
    
    # 创建 GUI 窗口
    gui = ti.GUI("3D Transformation (Taichi)", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 计算变换
        compute_transform(angle)
        
        # 获取全部 8 个顶点的 2D 坐标
        p = [screen_coords[i] for i in range(8)]
        
        # 1. 绘制前面 4 条边 (红色)
        gui.line(p[0], p[1], radius=2, color=0xFF0000)
        gui.line(p[1], p[2], radius=2, color=0xFF0000)
        gui.line(p[2], p[3], radius=2, color=0xFF0000)
        gui.line(p[3], p[0], radius=2, color=0xFF0000)
        
        # 2. 绘制后面 4 条边 (绿色)
        gui.line(p[4], p[5], radius=2, color=0x00FF00)
        gui.line(p[5], p[6], radius=2, color=0x00FF00)
        gui.line(p[6], p[7], radius=2, color=0x00FF00)
        gui.line(p[7], p[4], radius=2, color=0x00FF00)
        
        # 3. 绘制前后连接的 4 条侧面边 (蓝色)
        gui.line(p[0], p[4], radius=2, color=0x0000FF)
        gui.line(p[1], p[5], radius=2, color=0x0000FF)
        gui.line(p[2], p[6], radius=2, color=0x0000FF)
        gui.line(p[3], p[7], radius=2, color=0x0000FF)
        
        gui.show()

if __name__ == '__main__':
    main()