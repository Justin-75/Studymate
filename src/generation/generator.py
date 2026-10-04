# src/generation/generator.py
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from src.llm_client import generate, llm_enabled
from src.db.models import GenerateRequest, GenerateResult, SearchHit


# ----------------------------
# Citations (pages only)
# ----------------------------

def _build_citations_pages(hits: List[SearchHit], max_pages: int = 8) -> List[Dict[str, Any]]:
    pages: List[int] = []
    for h in sorted(hits or [], key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True):
        p = int(getattr(h, "page_no", 0) or 0)
        if p <= 0:
            continue
        if p in pages:
            continue
        pages.append(p)
        if len(pages) >= max_pages:
            break
    return [{"page_no": p} for p in pages]


# ----------------------------
# Public API
# ----------------------------


def _message(req: GenerateRequest, msg: str, citations: List[Dict[str, Any]] | None = None) -> GenerateResult:
    return GenerateResult(
        doc_id=req.doc_id,
        mode=req.mode,
        query=req.query,
        content={"message": msg},
        citations=citations or [],
    )


def _check_no_relevant_content(parsed: dict, req) -> GenerateResult | None:
    """If the LLM flagged the query as irrelevant, return a proper result."""
    if isinstance(parsed, dict) and parsed.get("no_relevant_content"):
        msg = parsed.get("message", "No relevant content found for the query.")
        return GenerateResult(
            doc_id=req.doc_id,
            mode=req.mode,
            query=req.query,
            content={"message": msg},
            citations=[],
        )
    return None


def build_summary_prompt(context: str, query: str) -> str:
    return f"""你是一个专业的学习助手，需要根据用户提供的文档片段和查询生成摘要。

**重要规则**：首先判断用户查询是否与文档内容相关。
- 如果用户的查询与文档内容完全无关（例如文档是关于算法，但用户问的是食物、天气等无关话题），
  你必须返回以下 JSON，不要强行生成不相关的内容：
  {{"no_relevant_content": true, "message": "The document does not contain information related to your query."}}
- 只有当查询确实与文档内容相关时，才生成摘要。
- 绝对不要用文档中的内容来回答一个与文档无关的查询。

文档片段：
{context}

用户查询：{query}

如果查询与文档相关，请生成一份结构清晰的摘要，包含：
1. 核心观点（一句话总结）
2. 关键要点（列表形式，每个要点标注来源页码，如果有页码信息）
3. 延伸解释（可选）

输出格式必须为纯 JSON，不要包含其他文字：
{{
  "main_idea": "核心观点",
  "key_points": [
    {{"text": "要点1", "page": 页码}},
    {{"text": "要点2", "page": 页码}}
  ],
  "summary": "完整摘要"
}}
"""


def build_flashcards_prompt(context: str, query: str, max_cards: int = 10) -> str:
    return f"""你是一个专业的学习助手，需要根据文档内容生成抽认卡。

**重要规则**：首先判断用户查询是否与文档内容相关。
- 如果用户的查询与文档内容完全无关（例如文档是关于算法，但用户问的是食物、天气等无关话题），
  你必须返回以下 JSON，不要强行生成不相关的内容：
  {{"no_relevant_content": true, "message": "The document does not contain information related to your query."}}
- 只有当查询确实与文档内容相关时，才生成抽认卡。
- 绝对不要用文档中的内容来回答一个与文档无关的查询。

文档内容：
{context}

根据以上文档内容，围绕「{query}」生成最多 {max_cards} 张学习抽认卡。
每张卡包含概念（front）、解释（back）、以及为什么重要（why）。
内容必须基于文档，不要编造。

请输出纯JSON (只输出 JSON，不要任何额外文字)，格式如下：
{{
  "cards": [
    {{
      "front": "概念或问题",
      "back": "解释或答案",
      "why": "为什么掌握这个知识点很重要"
    }}
  ]
}}
"""

def build_quiz_prompt(context: str, query: str, max_questions: int = 7) -> str:
    return f"""根据以下文档内容，围绕「{query}」生成 {max_questions} 道测验题。

**重要规则**：首先判断用户查询是否与文档内容相关。
- 如果用户的查询与文档内容完全无关（例如文档是关于算法，但用户问的是食物、天气等无关话题），
  你必须返回以下 JSON，不要强行生成不相关的内容：
  {{"no_relevant_content": true, "message": "The document does not contain information related to your query."}}
- 只有当查询确实与文档内容相关时，才生成测验题。
- 绝对不要用文档中的内容来回答一个与文档无关的查询。

混合三种题型：选择题(mcq)、填空题(fill_blank)、简答题(short)。
每道题标注难度："简单"、"中" 或 "难"。
内容必须基于文档，不要编造。

文档内容：
{context}

请输出纯JSON，格式如下：
{{
  "quiz": [
    {{
      "id": "q1",
      "type": "mcq",
      "difficulty": "简单",
      "question": "问题文本",
      "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
      "answer": "A"
    }},
    {{
      "id": "q2",
      "type": "fill_blank",
      "difficulty": "中",
      "question": "______ 是某个概念的定义。",
      "answer": "关键词"
    }},
    {{
      "id": "q3",
      "type": "short",
      "difficulty": "难",
      "question": "简述某个概念。",
      "answer": "简短的答案（1-2句话）"
    }}
  ]
}}
"""

def generate_material(req: GenerateRequest, hits: List[SearchHit]) -> GenerateResult:
    """生成学习材料（支持 LLM 增强）"""

    hits = hits or []

    if not hits:
        return GenerateResult(
            doc_id=req.doc_id,
            mode=req.mode,
            query=req.query,
            content={"message": "No relevant content found for the query."},
            citations=[],
        )
    citations = _build_citations_pages(hits)  # 复用原有 citations

    if not llm_enabled():
        return _message(req, "LLM is not configured. Set LLM_PROVIDER and its API key in .env (see src/llm_client.py).")

    # ----- LLM 生成 -----
    if hits:
        # 构建上下文：将所有 hits 的文本拼接，附上页码
        context_parts = []
        for hit in hits:
            page_info = f"[第 {hit.page_no} 页] {hit.text}"
            context_parts.append(page_info)
        context = "\n\n".join(context_parts)

        content = None
        llm_response = None

        if req.mode == "summary":
            prompt = build_summary_prompt(context, req.query or "")
            llm_response = generate(prompt, temperature=0.3)
            if llm_response:
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        content = json.loads(json_match.group())
                        # Check if LLM flagged as irrelevant
                        no_rel = _check_no_relevant_content(content, req)
                        if no_rel:
                            return no_rel
                        # 补充原有结构可能缺失的字段
                        content.setdefault("topics", [])
                        content.setdefault("concepts", [])
                        content.setdefault("pages", [])
                        content.setdefault("language", "cjk")  # 默认中文
                    except json.JSONDecodeError:
                        print("LLM 摘要 JSON 解析失败，原始响应:", llm_response)
                        content = None

        elif req.mode == "flashcards":
            prompt = build_flashcards_prompt(context, req.query or "")
            llm_response = generate(prompt, temperature=0.5)
            content = None
            if llm_response:
                # 尝试直接解析为 JSON
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        # Check if LLM flagged as irrelevant
                        no_rel = _check_no_relevant_content(parsed, req)
                        if no_rel:
                            return no_rel
                        if isinstance(parsed, dict) and "cards" in parsed:
                            content = parsed
                        elif isinstance(parsed, list):
                            # 如果直接返回数组，包装成对象
                            content = {"cards": parsed}
                        elif isinstance(parsed, dict) and len(parsed) == 0:
                            content = {"cards": []}
                        else:
                            # 可能是单个卡片对象
                            if "front" in parsed or "back" in parsed:
                                content = {"cards": [parsed]}
                    except json.JSONDecodeError:
                        content = None
                # 如果解析失败，尝试用正则提取每个对象并组装
                if not content:
                    # 匹配所有花括号对象
                    objects = re.findall(r'\{[^{}]*\}', llm_response)
                    if objects:
                        cards = []
                        for obj_str in objects:
                            try:
                                card = json.loads(obj_str)
                                if isinstance(card, dict):
                                    # 标准化字段名
                                    if "front" not in card:
                                        if "question" in card:
                                            card["front"] = card["question"]
                                        elif "term" in card:
                                            card["front"] = card["term"]
                                    if "back" not in card:
                                        if "answer" in card:
                                            card["back"] = card["answer"]
                                        elif "definition" in card:
                                            card["back"] = card["definition"]
                                    if "front" in card and "back" in card:
                                        cards.append(card)
                            except:
                                continue
                        if cards:
                            content = {"cards": cards}
            if content:
                print("✅ LLM 抽认卡生成成功")
                return GenerateResult(
                    doc_id=req.doc_id,
                    mode=req.mode,
                    query=req.query,
                    content=content,
                    citations=citations,
                )
            else:
                print("⚠️ LLM 抽认卡失败")

        elif req.mode == "quiz":
            prompt = build_quiz_prompt(context, req.query or "")
            llm_response = generate(prompt, temperature=0.3)
            content = None
            if llm_response:
                print("📝 LLM quiz raw:", llm_response)
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        # Check if LLM flagged as irrelevant
                        no_rel = _check_no_relevant_content(parsed, req)
                        if no_rel:
                            return no_rel
                        # 兼容 "quiz" 或 "questions"
                        quiz_list = parsed.get("quiz") or parsed.get("questions")
                        if quiz_list and isinstance(quiz_list, list):
                            # 标准化每个题目
                            for i, q in enumerate(quiz_list):
                                if "id" not in q:
                                    q["id"] = f"q{i}"
                                if "type" not in q:
                                    q["type"] = "mcq"
                                # 确保 difficulty 字段存在
                                if "difficulty" not in q:
                                    q["difficulty"] = "简单"
                                # MCQ 类型校验
                                if q["type"] == "mcq":
                                    if "answer" not in q:
                                        q["answer"] = "A"
                                    if "options" not in q or not isinstance(q.get("options"), list) or len(q["options"]) != 4:
                                        q["options"] = ["A. 是", "B. 否", "C. 不确定", "D. 不知道"]
                                # 填空题和简答题校验
                                elif q["type"] in ("fill_blank", "short"):
                                    if "answer" not in q:
                                        q["answer"] = ""
                            content = {"quiz": quiz_list}
                    except json.JSONDecodeError:
                        pass
            if content:
                # 校验每个题目
                valid_questions = []
                for q in content.get("quiz", []):
                    qtype = q.get("type", "mcq")
                    # 检查必要字段
                    if "id" not in q or "question" not in q or "answer" not in q:
                        continue
                    if qtype == "mcq":
                        # MCQ 需要 4 个选项，答案在 A-D
                        if not isinstance(q.get("options"), list) or len(q["options"]) != 4:
                            continue
                        if q["answer"] not in ("A", "B", "C", "D"):
                            continue
                    elif qtype in ("fill_blank", "short"):
                        # 填空/简答需要非空答案
                        if not str(q.get("answer", "")).strip():
                            continue
                    else:
                        continue  # 未知题型跳过
                    valid_questions.append(q)
                
                if valid_questions:
                    content["quiz"] = valid_questions
                    print(f"✅ 通过校验的题目数: {len(valid_questions)}")
                else:
                    content = None  # 没有有效题目
                    
        # 如果 LLM 成功生成了内容，则返回
        if content:
            return GenerateResult(
                doc_id=req.doc_id,
                mode=req.mode,
                query=req.query,
                content=content,
                citations=citations,
            )

    return _message(req, "Generation failed: the LLM returned no usable content. Please try again.", citations)
