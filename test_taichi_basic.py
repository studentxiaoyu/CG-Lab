import taichi as ti

print("开始测试 Taichi...")
ti.init(arch=ti.cpu)
print("Taichi 初始化成功")

x = ti.field(dtype=ti.f32, shape=5)
print("Taichi 字段创建成功")

@ti.kernel
def fill():
    for i in x:
        x[i] = i * 2.0

fill()
print("Taichi 内核执行成功")

print("结果:", x.to_numpy())
print("测试完成")