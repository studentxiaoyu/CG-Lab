import sys

print("开始测试 Taichi 导入...")
print(f"Python 版本: {sys.version}")

try:
    print("\n尝试导入 Taichi 模块...")
    import taichi
    print("Taichi 模块导入成功")
    print(f"Taichi 版本: {taichi.__version__}")
    
    print("\n尝试导入 Taichi 为 ti...")
    import taichi as ti
    print("Taichi 别名导入成功")
    
    print("\n测试完成")
    
except Exception as e:
    print(f"\n发生错误: {e}")
    import traceback
    traceback.print_exc()