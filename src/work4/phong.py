import taichi as ti
import config
from geometry import normalize, reflect, intersect_sphere, intersect_cone

@ti.kernel
def render_basic():
    for i, j in config.pixels:
        u = (i - config.res_x / 2.0) / config.res_y * 2.0
        v = (j - config.res_y / 2.0) / config.res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        color = ti.Vector([0.05, 0.15, 0.15]) 

        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            # 标准 Phong 光照计算 (不含阴影)
            ambient = config.Ka[None] * light_color * hit_color
            diffuse = config.Kd[None] * ti.max(0.0, N.dot(L)) * light_color * hit_color
            
            R = normalize(reflect(-L, N))
            specular = config.Ks[None] * (ti.max(0.0, R.dot(V)) ** config.shininess[None]) * light_color 
            
            color = ambient + diffuse + specular
                
        config.pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    config.init_default_parameters()
    window = ti.ui.Window("Basic Phong Shading", (config.res_x, config.res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render_basic()
        canvas.set_image(config.pixels)
        
        # 动态吸附右侧的 UI 逻辑
        p_width = 0.32
        p_height = 0.25
        p_x = 1.0 - p_width - 0.02 
        
        with gui.sub_window("Basic Parameters", p_x, 0.05, p_width, p_height):
            config.Ka[None] = gui.slider_float('Ka (Ambient)', config.Ka[None], 0.0, 1.0)
            config.Kd[None] = gui.slider_float('Kd (Diffuse)', config.Kd[None], 0.0, 1.0)
            config.Ks[None] = gui.slider_float('Ks (Specular)', config.Ks[None], 0.0, 1.0)
            config.shininess[None] = gui.slider_float('Shininess', config.shininess[None], 1.0, 128.0)

        window.show()

if __name__ == '__main__':
    main()