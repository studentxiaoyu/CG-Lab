import taichi as ti

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
aa_samples = ti.field(ti.i32, shape=()) # 新增：抗锯齿采样次数

# 材质常量枚举 (新增玻璃材质)
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """
    计算折射光线 (斯涅尔定律)
    I: 入射方向 (单位向量)
    N: 表面法线 (单位向量)
    eta: n1 / n2 (折射率之比)
    返回值: (折射方向, 是否发生折射)
    """
    cosi = -I.dot(N)
    cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi)
    out_dir = ti.Vector([0.0, 0.0, 0.0])
    is_refracted = False
    
    # 如果 cost2 > 0，说明没有发生全反射
    if cost2 > 0.0:
        out_dir = eta * I + (eta * cosi - ti.sqrt(cost2)) * N
        is_refracted = True
        
    return out_dir, is_refracted

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测左侧球：改为玻璃材质 (原本的红球)
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 1.0, 1.0]) # 纯净透明的玻璃色
        hit_mat = MAT_GLASS

    # 2. 检测右侧球：银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        color_sum = ti.Vector([0.0, 0.0, 0.0])
        
        # --- 选做：MSAA 抗锯齿采样循环 ---
        for sample in range(aa_samples[None]):
            # 在当前像素内部引入微小的随机偏移 (0.0 到 1.0 之间)
            u = (i + ti.random() - res_x / 2.0) / res_y * 2.0
            v = (j + ti.random() - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                if t > 1e9:
                    final_color += throughput * bg_color
                    break
                    
                p = ro + rd * t
                
                # --- 选做：玻璃材质处理 ---
                if mat_id == MAT_GLASS:
                    ior = 1.5 # 玻璃的折射率 (Index of Refraction)
                    cosi = rd.dot(N)
                    
                    out_n = N
                    eta = 1.0 / ior # 默认从空气进入玻璃
                    
                    # 判断光线是在球外面还是在球里面
                    if cosi > 0.0:
                        out_n = -N # 光线正在从球体内部射出，法线反向
                        eta = ior / 1.0 # 从玻璃回到空气
                        
                    # 计算折射
                    refract_dir, is_refracted = refract(rd, out_n, eta)
                    
                    if is_refracted:
                        # 成功折射：光线穿过表面
                        ro = p - out_n * 1e-4 # 核心避坑：沿着法线反向缩进一点点，确保进入内部
                        rd = normalize(refract_dir)
                    else:
                        # 发生全反射 (Total Internal Reflection)：光线出不去，被弹回内部
                        ro = p + out_n * 1e-4
                        rd = normalize(reflect(rd, out_n))
                        
                    throughput *= obj_color 
                    # 玻璃不阻挡光线，继续循环
                    
                # 分支 2：镜面反射材质
                elif mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color 
                    
                # 分支 3：漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                    
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0 
                        
                    ambient = 0.2 * obj_color
                    direct_light = ambient 
                    
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse
                    
                    final_color += throughput * direct_light
                    break
                    
            # 累加单次采样的颜色
            color_sum += final_color

        # 将采样结果求平均值，并进行钳制
        pixels[i, j] = ti.math.clamp(color_sum / float(aa_samples[None]), 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo - Bonus", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化参数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    aa_samples[None] = 4 # 默认开启 4x 抗锯齿

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.70, 0.05, 0.28, 0.28): # 面板稍微拉大一点点
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            # 添加抗锯齿的 UI 滑动条
            aa_samples[None] = gui.slider_int('AA Samples (MSAA)', aa_samples[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()