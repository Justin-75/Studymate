import sys
import os
import traceback
import tempfile
import uuid
from pathlib import Path

# ========== 绝对路径设置 ==========
# 获取项目根目录的绝对路径
project_root = Path(r"E:\PersonalProject\studymate").resolve()
print(f"项目根目录（绝对路径）: {project_root}")
print(f"当前工作目录: {os.getcwd()}")

# 确保项目根目录在Python路径的最前面
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"已添加项目根目录到Python路径")

print("Python路径前5个:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i+1}. {path}")

print("="*60)

# ========== 导入项目模块 ==========
print("正在导入项目模块...")

try:
    from src.ingest import ingest_pdf
    print("✅ 导入 ingest_pdf")
except ImportError as e:
    print(f"❌ 导入 ingest_pdf 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.generation.generator import generate_material
    print("✅ 导入 generate_material")
except ImportError as e:
    print(f"❌ 导入 generate_material 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.generation.quiz_grader import grade_quiz
    print("✅ 导入 grade_quiz")
except ImportError as e:
    print(f"❌ 导入 grade_quiz 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.db.models import GenerateRequest
    print("✅ 导入 GenerateRequest")
except ImportError as e:
    print(f"❌ 导入 GenerateRequest 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.db.repository import Repository
    print("✅ 导入 Repository")
except ImportError as e:
    print(f"❌ 导入 Repository 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.retrieval.hybrid import hybrid_search, warm_up
    print("✅ 导入 hybrid_search (BM25 + BGE-M3 + RRF + reranker)")
except ImportError as e:
    print(f"❌ 导入 hybrid_search 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("="*60)
print("✅ 所有模块导入成功！")
print("="*60)

# ========== 导入Flask ==========
from flask import Flask, request, jsonify
from flask_cors import CORS

# ========== 创建Flask应用 ==========
app = Flask(__name__)
CORS(app)

# ========== 全局配置 ==========
DB_PATH = str(project_root / "Data/Database/app.db")

print(f"数据库路径: {DB_PATH}")

# ========== 上传配置 ==========
UPLOAD_FOLDER = project_root / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
print(f"上传目录: {UPLOAD_FOLDER}")

# ========== 路由 ==========
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'ok', 
        'message': '服务器运行正常',
        'endpoints': {
            'POST /api/upload': '上传PDF文件',
            'POST /api/generate': '生成学习材料',
            'POST /api/grade': '评分测验',
            'GET /api/test': '测试连接'
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """上传PDF"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存临时文件
    temp_filename = f"{uuid.uuid4()}.pdf"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    file.save(temp_path)
    
    try:
        print(f"开始处理PDF: {temp_path}")
        
        # 调用ingest_pdf，传递必要的参数
        result = ingest_pdf(
            file_path=temp_path,
            db_path=DB_PATH,
            ocr_enabled=True,
            ocr_lang="eng+chi_sim",
            ocr_dpi=300,
            ocr_psm=3,
            ocr_preprocess=False,
            chunk_size=800,
            chunk_overlap=120,
            max_pages=None
        )
        
        print(f"ingest_pdf返回结果: {result}")
        
        # 提取doc_id
        doc_id = None
        if hasattr(result, 'doc_id'):
            doc_id = result.doc_id
        elif isinstance(result, dict) and 'doc_id' in result:
            doc_id = result['doc_id']
        elif isinstance(result, str):
            doc_id = result
        else:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Build the BM25 + dense indexes now, so the first question doesn't wait for them
        print(f"构建检索索引 (BM25 + BGE-M3): {doc_id}")
        warm_up(Repository(DB_PATH), doc_id)
        
        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'message': 'PDF处理成功'
        })
    except Exception as e:
        print(f"处理PDF失败: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """生成学习材料"""
    print("收到生成请求")
    
    data = request.json
    if not data:
        return jsonify({'error': '没有数据'}), 400
    
    required = ['doc_id', 'mode']
    for field in required:
        if field not in data:
            return jsonify({'error': f'缺少字段: {field}'}), 400
    
    try:
        # 创建GenerateRequest
        from enum import Enum
        
        class GenerateMode(Enum):
            SUMMARY = "summary"
            FLASHCARDS = "flashcards"
            QUIZ = "quiz"
        
        doc_id = data['doc_id']
        mode = data['mode']
        query = data.get('query', '')
        top_k = data.get('top_k', 8)
        
        print(f"生成参数: doc_id={doc_id}, mode={mode}, query={query}, top_k={top_k}")
        
        repo = Repository(DB_PATH)

        # Hybrid retrieval: BM25 + BGE-M3 dense -> RRF -> bge-reranker (same code as the benchmark)
        print(f"开始检索: doc_id={doc_id}, query={query}")
        hits = hybrid_search(repo, doc_id, query, top_k=top_k)
        print(f"检索到 {len(hits)} 个结果")
        
        # 创建GenerateRequest对象
        req = GenerateRequest(
            doc_id=doc_id,
            mode=mode,
            query=query,
            top_k=top_k
        )
        
        # 调用generate_material
        print(f"开始生成: mode={mode}")
        result = generate_material(req, hits)
        print(f"生成完成，结果类型: {type(result)}")
        
        # 准备响应数据
        response_data = {
            'success': True,
            'data': result.content
        }
        
        # 如果是测验模式，提取quiz_id
        if mode == 'quiz' and hasattr(result, 'content'):
            if isinstance(result.content, dict) and 'quiz_id' in result.content:
                response_data['quiz_id'] = result.content['quiz_id']
                print(f"提取到quiz_id: {response_data['quiz_id']}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"生成失败: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/grade', methods=['POST'])
def grade():
    """评分测验"""
    print("收到评分请求")
    
    data = request.json
    if not data:
        return jsonify({'error': '没有数据'}), 400
    
    required = ['quiz', 'answers']
    for field in required:
        if field not in data:
            return jsonify({'error': f'缺少字段: {field}'}), 400
    
    try:
        quiz = data['quiz']
        answers = data['answers']
        
        print(f"开始评分: quiz_id={quiz.get('quiz_id', 'unknown')}")
        result = grade_quiz(quiz, answers)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        print(f"评分失败: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ========== 主程序 ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("📚 PDF学习伴侣服务器启动")
    print(f"访问: http://localhost:5000/api/test")
    print("="*60)
    
    # 禁用重新加载，避免导入问题
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)