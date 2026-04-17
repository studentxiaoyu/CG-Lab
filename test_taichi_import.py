print("开始测试 Taichi 导入...")

try:
    import taichi as ti
    print("Taichi 导入成功")
    
    print(f"Taichi 版本: {ti.__version__}")
    
    print("正在初始化 Taichi...")
    ti.init(arch=ti.cpu)
    print("Taichi 初始化成功")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")