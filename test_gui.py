import taichi as ti

ti.init(arch=ti.cpu)

print("正在创建 GUI 窗口...")
gui = ti.GUI("Test Window", res=(400, 400))

print("GUI 窗口已创建，请查看弹窗")

frame_count = 0
while gui.running:
    gui.text("Hello Taichi!", pos=(0.5, 0.5), color=0xFFFFFF)
    gui.show()
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已运行 {frame_count} 帧")

print("程序结束")