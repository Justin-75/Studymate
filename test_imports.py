# test_imports.py - 测试导入
import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print(f"项目根目录: {project_root}")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径:")
for p in sys.path:
    print(f"  {p}")

print("\n" + "="*50 + "\n")

# 测试导入
modules_to_test = [
    "src.ingest",
    "src.generation.generator",
    "src.generation.quiz_grader",
    "src.db.models",
    "src.db.repository",
    "src.retrieval.search",
    "src.retrieval.tfidf_index"
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"✅ {module_name} 导入成功")
    except ImportError as e:
        print(f"❌ {module_name} 导入失败: {e}")