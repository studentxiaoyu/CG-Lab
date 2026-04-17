import sys
import traceback

print("开始调试 Taichi 程序...")
print(f"Python 版本: {sys.version}")
print(f"Python 可执行文件: {sys.executable}")

try:
    print("\n尝试导入 Taichi...")
    import taichi as ti
    print("Taichi 导入成功")
    print(f"Taichi 版本: {ti.__version__}")
    
    print("\n尝试初始化 Taichi...")
    ti.init(arch=ti.cpu, print_ir=False)
    print("Taichi 初始化成功")
    
    print("\n尝试创建 GUI 窗口...")
    gui = ti.GUI("Debug Window", res=(400, 400))
    print("GUI 窗口创建成功")
    
    print("\n尝试运行主循环...")
    frame_count = 0
    max_frames = 100  # 限制帧数，避免无限循环
    
    while gui.running and frame_count < max_frames:
        gui.text("Debug", pos=(0.5, 0.5), color=0xFFFFFF)
        gui.show()
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"已运行 {frame_count} 帧")
    
    print(f"\n主循环结束，共运行 {frame_count} 帧")
    
except Exception as e:
    print(f"\n发生错误: {e}")
    print("错误详情:")
    traceback.print_exc()
    
print("\n调试完成")