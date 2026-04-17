import taichi as ti
import numpy as np

# 使用 gpu 后端获取最佳性能
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000 # 贝塞尔曲线的采样数
MAX_CURVE_POINTS = 5000 # 显存池的极限容量，防止点数过多溢出

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 用于存放曲线坐标的固定大小 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)


# ==========================================================
# 算法核心区
# ==========================================================

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法 (用于贝塞尔曲线)"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

def compute_cubic_bspline(points, total_segments):
    """【选做 2】使用基矩阵实现均匀三次 B 样条曲线计算"""
    n = len(points)
    if n < 4:
        return np.zeros((0, 2), dtype=np.float32)
    
    num_curves = n - 3
    # 动态分配每段的采样数，确保总点数可控
    segments_per_curve = max(10, total_segments // num_curves)
    
    # 三次 B 样条的固定基矩阵 (Basis Matrix)
    M = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  0,  3,  0],
        [ 1,  4,  1,  0]
    ], dtype=np.float32) / 6.0
    
    pts = []
    for i in range(num_curves):
        # 提取当前分段的 4 个局部控制点
        P = np.array([points[i], points[i+1], points[i+2], points[i+3]], dtype=np.float32)
        
        # 利用矩阵乘法求解曲线上的局部坐标
        for j in range(segments_per_curve + 1):
            t = j / segments_per_curve
            T = np.array([t**3, t**2, t, 1.0], dtype=np.float32)
            pt = T @ M @ P  # 多项式矩阵求值
            pts.append(pt)
            
    return np.array(pts, dtype=np.float32)


# ==========================================================
# GPU 渲染管线
# ==========================================================

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    """【选做 1】改进的光栅化绘制：支持亚像素精度的平滑反走样"""
    for i in range(n):
        pt = curve_points_field[i]
        
        # 精确的几何浮点坐标
        x_exact = pt[0] * WIDTH
        y_exact = pt[1] * HEIGHT
        
        # 所在的基准像素网格
        x_base = ti.cast(x_exact, ti.i32)
        y_base = ti.cast(y_exact, ti.i32)
        
        # 考察周围 3x3 的邻域像素
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = x_base + dx
                py = y_base + dy
                
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # 计算该像素中心点 (px+0.5, py+0.5) 与精确几何坐标之间的空间距离
                    dist = ti.sqrt((px + 0.5 - x_exact)**2 + (py + 0.5 - y_exact)**2)
                    
                    # 距离衰减模型：采用类高斯衰减。距离越近，权重越高 (平滑过渡核心)
                    weight = ti.exp(-dist * dist * 1.5)
                    
                    # 以加法混色模式（Additive Blending）叠加颜色，并限制上限为 1.0 避免过曝
                    pixels[px, py][0] = ti.min(1.0, pixels[px, py][0] + r * weight)
                    pixels[px, py][1] = ti.min(1.0, pixels[px, py][1] + g * weight)
                    pixels[px, py][2] = ti.min(1.0, pixels[px, py][2] + b * weight)


# ==========================================================
# 主程序与交互系统
# ==========================================================

def main():
    window = ti.ui.Window("Graphics Curve (Anti-Aliasing & B-Spline)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    mode = 'bezier'  # 初始模式
    print("=== 控制台信息 ===")
    print("当前模式: 贝塞尔曲线 (绿色)")
    print("[点击左键] 添加控制点 | [C] 清空画布 | [B] 切换曲线模式")
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c': 
                control_points = []
            elif e.key == 'b':  # 切换渲染模式
                mode = 'bspline' if mode == 'bezier' else 'bezier'
                print(f"模式已切换为: {'B 样条曲线 (橙色)' if mode == 'bspline' else '贝塞尔曲线 (绿色)'}")
        
        clear_pixels()
        
        current_count = len(control_points)
        curve_points_np = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)
        valid_points = 0
        
        # 1. 贝塞尔曲线模式
        if mode == 'bezier' and current_count >= 2:
            valid_points = NUM_SEGMENTS + 1
            for t_int in range(valid_points):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(valid_points, 0.0, 1.0, 0.0) # 绿色表示贝塞尔
            
        # 2. B 样条曲线模式 (要求点数至少为 4)
        elif mode == 'bspline' and current_count >= 4:
            bspline_np = compute_cubic_bspline(control_points, NUM_SEGMENTS)
            valid_points = min(len(bspline_np), MAX_CURVE_POINTS)
            
            curve_points_np[:valid_points] = bspline_np[:valid_points]
            
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(valid_points, 1.0, 0.6, 0.0) # 橙色表示B样条
                    
        canvas.set_image(pixels)
        
        # 绘制背景的控制点和折线
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.4, 0.4, 0.4))
        
        window.show()

if __name__ == '__main__':
    main()