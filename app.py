import streamlit as st
import google.generativeai as genai
from typing import Dict, List
import json
from datetime import datetime
import ast
import re
import os
from dotenv import load_dotenv
from tavily import TavilyClient  # 新增导入，连网搜索

# 加载 .env 文件（如果存在）
load_dotenv()

# ==================== 配置区域 ====================
# 从环境变量或 Streamlit secrets 读取 API Key
# 优先级：st.secrets > 环境变量 > 空字符串
def get_api_key() -> str:
    """获取 Gemini API Key，支持多种来源"""
    # 1. 优先从 Streamlit secrets 读取（用于 Streamlit Cloud 部署）
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except Exception:
        pass
    
    # 2. 从环境变量读取（支持 .env 文件）
    api_key = os.getenv('GEMINI_API_KEY', '')
    if api_key:
        return api_key
    
    # 3. 如果都没有，返回空字符串
    return ""


# 获取 Tavily API Key
def get_tavily_key() -> str:
    if hasattr(st, 'secrets') and 'TAVILY_API_KEY' in st.secrets:
        return st.secrets['TAVILY_API_KEY']
    return os.getenv('TAVILY_API_KEY', '')

TAVILY_API_KEY = get_tavily_key()
GEMINI_API_KEY = get_api_key()

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="PM Insight Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式定制 ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analysis-section {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    /* 让 tabs 均匀分布 */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        gap: 0;
        justify-content: space-around;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 初始化 Gemini ====================
def init_gemini():
    """初始化 Gemini API"""
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
        st.error("⚠️ 请配置您的 Gemini API Key")
        st.info("""
        **配置方式（任选一种）：**
        
        1. **本地开发**：创建 `.env` 文件，添加：
           ```
           GEMINI_API_KEY=your_api_key_here
           ```
        
        2. **Streamlit Cloud**：在 Streamlit Cloud 的 Secrets 中添加：
           ```
           GEMINI_API_KEY=your_api_key_here
           ```
        
        3. **环境变量**：直接设置环境变量 `GEMINI_API_KEY`
        """)
        st.stop()
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 配置生成参数：强制 JSON 输出，增加最大输出 token 数
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=8192,  # 增加到 8192 以避免 JSON 截断
            response_mime_type="application/json"
        )
        
        return genai.GenerativeModel(
            "gemini-2.5-flash-lite",
            generation_config=generation_config
        )
    except Exception as e:
        st.error(f"❌ Gemini API 初始化失败: {str(e)}")
        st.stop()

def fetch_competitor_context(product_input: str) -> str:
    """使用 Tavily 获取竞品的实时市场信息"""
    if not TAVILY_API_KEY:
        return "（未配置 Tavily API，使用模型内置知识分析）"
    
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        # 构造搜索词：竞品名 + 最新功能 + 用户评价 + 融资情况
        search_query = f"{product_input} latest features user feedback and market position 2025"
        
        # 执行高级搜索，获取前 5 条深度内容
        search_result = tavily.search(query=search_query, search_depth="advanced", max_results=5)
        
        context = "以下是从互联网搜集的实时信息：\n"
        for i, res in enumerate(search_result['results'], 1):
            context += f"资料[{i}]: {res['content'][:1000]}\n来源: {res['url']}\n\n"
        return context
    except Exception as e:
        return f"（搜索执行失败: {str(e)}）"
    

# ==================== 分析提示词模板 ====================
def create_analysis_prompt(product_input: str, web_context: str = "") -> str:
    """创建强迫结构化输出且 Key 严格对齐的分析提示词"""
    prompt = f"""
你是一位在硅谷深耕多年的资深 AI 产品战略专家。
请结合以下【实时搜集的情报】，对竞品 '{product_input}' 进行深度拆解。

【实时情报参考】
{web_context}

**🎯 核心输出要求（必须严格遵守）：**
你必须返回一个严格的 JSON 对象，且 JSON 的 Key 必须【完全匹配】以下定义的名称，不得有误：

1. **Key: "model_stack"**
   内容要求：分析技术底座、AI 模型依赖、技术瓶颈。使用 Markdown 的 `###` 标题和列表。

2. **Key: "scene_fit"**
   内容要求：分析核心场景、用户准入门槛、场景延展性。使用 Markdown 的 `###` 标题和列表。

3. **Key: "data_moat"**
   内容要求：分析数据获取、反馈飞轮、护城河可持续性。使用 Markdown 的 `###` 标题和列表。

4. **Key: "ux_friction"**
   内容要求：分析认知负担、交互摩擦点、体验改进建议。使用 Markdown 的 `###` 标题和列表。

5. **Key: "commercial_roi"**
   内容要求：分析变现引擎、成本收益推算、增长潜力。使用 Markdown 的 `###` 标题和列表。

6. **Key: "strategy_advice"**
   内容要求：给出 1-2 条具体的、加粗的错位竞争金句建议。

**⚠️ 格式禁令：**
- **JSON 字段名（Key）严禁包含数字前缀**（如不要写成 "1. model_stack"）。
- 所有 Value 中的内容必须结构化，多用 **加粗** 和 列表。
- 所有的换行符必须转义为 '\\n'。
"""
    return prompt

# ==================== 执行分析 ====================
def perform_analysis(model, product_input: str, web_context: str = "") -> Dict:
    """执行竞品分析（回归 Single-shot JSON 模式）"""
    # 注入联网情报
    prompt = create_analysis_prompt(product_input, web_context)
    
    try:
        with st.spinner("🔍 正在进行深度建模与 JSON 构建..."):
            response = model.generate_content(prompt)
            response_text = response.text
        
        # 1. 基础清理
        json_text = response_text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:].strip()
        if json_text.startswith("```"):
            json_text = json_text[3:].strip()
        if json_text.endswith("```"):
            json_text = json_text[:-3].strip()
            
        try:
            # 2. 尝试标准解析
            analysis_result = json.loads(json_text)
        except json.JSONDecodeError:
            # 3. 失败时调用你写的 fix_truncated_json
            st.warning("⚠️ 检测到 JSON 异常，正在启动逻辑修复...")
            json_text_fixed = fix_truncated_json(json_text)
            analysis_result = json.loads(json_text_fixed)
        
        # 4. 字段验证与文本清理
        required_keys = ["model_stack", "scene_fit", "data_moat", "ux_friction", "commercial_roi", "strategy_advice"]
        for key in required_keys:
            if key not in analysis_result:
                analysis_result[key] = "内容生成异常"
            else:
                analysis_result[key] = clean_text(analysis_result[key])
                
        return analysis_result

    except Exception as e:
        st.error(f"❌ 分析失败: {str(e)}")
        # 即使彻底失败，也返回一个空结构防止前端崩溃
        return {k: f"分析失败: {str(e)}" for k in ["model_stack", "scene_fit", "data_moat", "ux_friction", "commercial_roi", "strategy_advice"]}

# ==================== 文本解析备用方案 ====================
def parse_text_response(text: str) -> Dict:
    """当 JSON 解析失败时，尝试从文本中提取结构化内容"""
    result = {
        "model_stack": "",
        "scene_fit": "",
        "data_moat": "",
        "ux_friction": "",
        "commercial_roi": "",
        "strategy_advice": ""
    }
    
    # 尝试按关键词提取
    sections = {
        "model_stack": ["Model Stack", "技术栈", "模型依赖"],
        "scene_fit": ["Scene-Fit", "场景", "细分场景"],
        "data_moat": ["Data Moat", "数据闭环", "护城河"],
        "ux_friction": ["UX Friction", "交互痛点", "用户体验"],
        "commercial_roi": ["Commercial ROI", "商业化", "价值评估"],
        "strategy_advice": ["错位竞争", "竞争建议", "差异化"]
    }
    
    for key, keywords in sections.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                # 简单提取：找到关键词后的内容
                idx = text.lower().find(keyword.lower())
                if idx != -1:
                    # 提取该段落（到下一个关键词或段落结束）
                    section_text = text[idx:idx+500]  # 提取500字符
                    result[key] = section_text
                    break
    
    # 如果都为空，返回原始文本
    if not any(result.values()):
        result["model_stack"] = text[:500]
        result["scene_fit"] = text[500:1000] if len(text) > 500 else ""
        result["data_moat"] = text[1000:1500] if len(text) > 1000 else ""
        result["ux_friction"] = text[1500:2000] if len(text) > 1500 else ""
        result["commercial_roi"] = text[2000:2500] if len(text) > 2000 else ""
        result["strategy_advice"] = "请查看完整分析内容"
    
    return result

# ==================== JSON 修复 ====================
def fix_truncated_json(json_text: str) -> str:
    """尝试修复被截断的 JSON"""
    json_text = json_text.strip()
    
    # 如果 JSON 为空或只有 {，返回一个基本的 JSON 结构
    if not json_text or json_text == '{':
        required_keys = ["model_stack", "scene_fit", "data_moat", "ux_friction", "commercial_roi", "strategy_advice"]
        return '{\n    ' + ',\n    '.join([f'"{key}": "内容被截断"' for key in required_keys]) + '\n}'
    
    # 检查是否在字符串中间被截断
    # 找到最后一个完整的字段
    last_comma = json_text.rfind(',')
    last_colon = json_text.rfind(':')
    
    # 如果最后一个字符是 :，说明字段值未完成
    if json_text.rstrip().endswith(':'):
        # 移除未完成的字段，从上一个完整字段开始
        if last_comma > 0:
            json_text = json_text[:last_comma + 1]
        else:
            # 如果没有逗号，说明这是第一个字段，需要移除
            json_text = json_text[:json_text.rfind('"', 0, last_colon) + 1] if last_colon > 0 else json_text
    
    # 处理未闭合的字符串
    # 计算未转义的引号数量
    quote_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(json_text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            if in_string:
                quote_count += 1
    
    # 如果字符串未闭合，尝试闭合
    if in_string:
        # 找到最后一个引号的位置，在其后添加闭合引号
        last_quote = json_text.rfind('"')
        if last_quote >= 0:
            # 检查是否需要转义最后一个字符
            if last_quote > 0 and json_text[last_quote - 1] == '\\':
                # 最后一个引号被转义了，需要添加新的引号
                json_text = json_text[:last_quote + 1] + '"'
            else:
                # 字符串应该已经闭合，但可能缺少值
                pass
    
    # 确保 JSON 结构完整
    open_braces = json_text.count('{')
    close_braces = json_text.count('}')
    
    # 移除末尾可能的未完成内容
    json_text = json_text.rstrip()
    
    # 如果最后一个字符不是 } 或 "，尝试修复
    if not json_text.endswith(('}', '"', ',')):
        # 尝试找到最后一个完整的字段值
        # 简单处理：如果以引号结尾，添加逗号和闭合括号
        if json_text.endswith('"'):
            # 检查是否需要添加逗号
            if open_braces > close_braces:
                json_text += '\n' + '}' * (open_braces - close_braces)
        else:
            # 可能字符串未闭合，尝试闭合
            json_text += '"'
            if open_braces > close_braces:
                json_text += '\n' + '}' * (open_braces - close_braces)
    
    # 检查必需字段
    required_keys = ["model_stack", "scene_fit", "data_moat", "ux_friction", "commercial_roi", "strategy_advice"]
    missing_keys = [key for key in required_keys if f'"{key}"' not in json_text]
    
    # 如果有缺失字段，添加它们
    if missing_keys:
        # 移除最后的 }，添加缺失字段，然后重新闭合
        json_text = json_text.rstrip().rstrip('}').rstrip(',').rstrip()
        if json_text.endswith('"'):
            json_text += ',\n'
        else:
            json_text += ',\n'
        
        for i, key in enumerate(missing_keys):
            json_text += f'    "{key}": "内容被截断"'
            if i < len(missing_keys) - 1:
                json_text += ',\n'
            else:
                json_text += '\n'
        
        json_text += '}'
    
    # 最后确保 JSON 以 } 结尾
    if not json_text.rstrip().endswith('}'):
        json_text = json_text.rstrip().rstrip(',')
        json_text += '\n}'
    
    return json_text

# ==================== 文本清理 ====================
def clean_text(text: str) -> str:
    """清理文本中的转义字符，转换为可读格式"""
    if not text or not isinstance(text, str):
        return text
    
    # 针对 JSON 字符串中的 Markdown 换行进行深度清理
    text = text.replace('\\\\n', '\n')
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '    ')
    
    # 清理多余的引号和首尾空格
    text = text.strip().strip('"')
    
    # 确保 Markdown 标题前有换行，防止渲染问题
    text = re.sub(r'([^\n])###', r'\1\n###', text)
    
    return text

# ==================== 历史记录管理 ====================
def add_to_history(product_name: str, analysis_result: Dict):
    """添加产品到历史记录，包含分析结果"""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # 如果已存在，先移除（避免重复）
    st.session_state['history'] = [h for h in st.session_state['history'] if h['product'] != product_name]
    
    # 添加到开头，保存完整的分析结果
    st.session_state['history'].insert(0, {
        'product': product_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'analysis_result': analysis_result  # 保存完整的分析结果
    })
    
    # 只保留最近 20 条记录
    if len(st.session_state['history']) > 20:
        st.session_state['history'] = st.session_state['history'][:20]

def get_history() -> List[Dict]:
    """获取历史记录"""
    return st.session_state.get('history', [])

def load_history_item(product_name: str) -> Dict:
    """从历史记录中加载指定产品的分析结果"""
    history = get_history()
    for item in history:
        if item['product'] == product_name:
            return item.get('analysis_result', {})
    return {}

# ==================== 格式化错位竞争建议 ====================
def format_competitive_advantage(competitive_advantage) -> str:
    """格式化错位竞争建议，将 JSON/Python 格式转换为可读文本"""
    if not competitive_advantage:
        return ""
    
    # 如果是字符串，尝试解析为 JSON 或 Python 字面量
    if isinstance(competitive_advantage, str):
        parsed = None
        
        # 方法1: 尝试解析为 JSON
        try:
            if competitive_advantage.strip().startswith(("[", "{")):
                parsed = json.loads(competitive_advantage)
            elif "[" in competitive_advantage or "{" in competitive_advantage:
                # 尝试提取 JSON 部分
                start_idx = competitive_advantage.find("[")
                if start_idx == -1:
                    start_idx = competitive_advantage.find("{")
                if start_idx != -1:
                    # 找到匹配的结束括号
                    bracket_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(competitive_advantage[start_idx:], start_idx):
                        if char in ['[', '{']:
                            bracket_count += 1
                        elif char in [']', '}']:
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                    json_str = competitive_advantage[start_idx:end_idx]
                    parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 方法2: 如果 JSON 解析失败，尝试使用 ast.literal_eval（更安全，可处理 Python 字面量）
        if parsed is None:
            try:
                if "[" in competitive_advantage or "{" in competitive_advantage:
                    start_idx = competitive_advantage.find("[")
                    if start_idx == -1:
                        start_idx = competitive_advantage.find("{")
                    if start_idx != -1:
                        # 找到匹配的结束括号
                        bracket_count = 0
                        end_idx = start_idx
                        for i, char in enumerate(competitive_advantage[start_idx:], start_idx):
                            if char in ['[', '{']:
                                bracket_count += 1
                            elif char in [']', '}']:
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break
                        python_str = competitive_advantage[start_idx:end_idx]
                        parsed = ast.literal_eval(python_str)
            except (ValueError, SyntaxError):
                # 如果都解析失败，返回原字符串
                return competitive_advantage
        
        if parsed is None:
            return competitive_advantage
    else:
        parsed = competitive_advantage
    
    # 格式化输出
    formatted_text = ""
    
    # 如果是列表
    if isinstance(parsed, list):
        for idx, item in enumerate(parsed, 1):
            if isinstance(item, dict):
                strategy = item.get("strategy", "")
                description = item.get("description", "")
                
                if strategy:
                    formatted_text += f"**策略 {idx}：{strategy}**\n\n"
                if description:
                    # 处理描述中的换行符和转义字符
                    description = description.replace("\\n", "\n")
                    # 清理多余的空白行
                    description = "\n".join(line.strip() for line in description.split("\n") if line.strip())
                    formatted_text += f"{description}\n\n"
                if idx < len(parsed):
                    formatted_text += "---\n\n"
            elif isinstance(item, str):
                formatted_text += f"**建议 {idx}：** {item}\n\n"
                if idx < len(parsed):
                    formatted_text += "---\n\n"
            else:
                formatted_text += f"{item}\n\n"
    
    # 如果是字典
    elif isinstance(parsed, dict):
        strategy = parsed.get("strategy", "")
        description = parsed.get("description", "")
        
        if strategy:
            formatted_text += f"**策略：{strategy}**\n\n"
        if description:
            description = description.replace("\\n", "\n")
            description = "\n".join(line.strip() for line in description.split("\n") if line.strip())
            formatted_text += f"{description}\n\n"
    
    # 如果是其他类型，转换为字符串
    else:
        formatted_text = str(parsed)
    
    return formatted_text.strip()

# ==================== Markdown 导出 ====================
def generate_markdown_report(product_name: str, analysis_result: Dict) -> str:
    """生成 Markdown 格式的报告"""
    markdown = f"""# 竞品分析报告：{product_name}

**生成时间：** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 执行摘要

本报告对 **{product_name}** 进行了深度竞品分析，从技术栈、场景适配、数据护城河、用户体验和商业化价值五个维度进行了全面评估。

---

## 1. 🔧 Model Stack（技术栈与模型依赖）

{analysis_result.get("model_stack", "暂无数据")}

---

## 2. 🎯 Scene-Fit（核心解决的细分场景）

{analysis_result.get("scene_fit", "暂无数据")}

---

## 3. 🛡️ Data Moat（数据闭环与护城河）

{analysis_result.get("data_moat", "暂无数据")}

---

## 4. ⚡ UX Friction（交互痛点分析）

{analysis_result.get("ux_friction", "暂无数据")}

---

## 5. 💰 Commercial ROI（商业化价值评估）

{analysis_result.get("commercial_roi", "暂无数据")}

---

## 💡 错位竞争建议

{format_competitive_advantage(analysis_result.get("strategy_advice", "")) if analysis_result.get("strategy_advice") else "暂无数据"}

---

*本报告由 PM Insight Copilot 自动生成*
"""
    return markdown

# ==================== 主界面 ====================
def main():
    # 侧边栏 - 历史记录
    with st.sidebar:
        st.header("📚 历史记录")
        history = get_history()
        
        if history:
            st.write(f"最近查询了 {len(history)} 个产品")
            st.markdown("---")
            
            # 显示历史记录列表
            for idx, record in enumerate(history):
                if st.button(f"{idx + 1}. {record['product']}", key=f"history_{idx}", use_container_width=True):
                    # 点击历史记录时，恢复分析结果
                    st.session_state['selected_product'] = record['product']
                    if 'analysis_result' in record:
                        st.session_state['last_analysis'] = record['analysis_result']
                        st.session_state['last_product'] = record['product']
                        # 生成 markdown 报告
                        markdown_report = generate_markdown_report(record['product'], record['analysis_result'])
                        st.session_state['last_markdown'] = markdown_report
                    st.rerun()
            
            # 清空历史记录按钮
            if st.button("🗑️ 清空历史记录", use_container_width=True):
                st.session_state['history'] = []
                st.rerun()
        else:
            st.info("暂无历史记录")
            st.write("开始分析后，查询记录将显示在这里")
    
    # 标题区域
    st.markdown('<h1 class="main-header">📊 PM Insight Copilot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI 驱动的竞品深度分析工具</p>', unsafe_allow_html=True)
    
    # 初始化 Gemini
    model = init_gemini()
    
    # 输入区域
    st.markdown("---")
    col_input1, col_input2 = st.columns([4, 1])
    
    with col_input1:
        # 如果从历史记录选择了产品，自动填充
        default_value = st.session_state.get('selected_product', '')
        # 如果没有从历史记录选择，使用上次分析的产品名称
        if not default_value:
            default_value = st.session_state.get('last_product', '')
        
        if st.session_state.get('selected_product'):
            # 使用后清除，避免下次还显示
            del st.session_state['selected_product']
        
        product_input = st.text_input(
            "请输入竞品名称或产品描述",
            value=default_value,
            placeholder="例如：ChatGPT、Notion AI、Midjourney 等",
            label_visibility="visible"
        )
    
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
        analyze_button = st.button("🚀 开始深度分析", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # 执行分析
    if analyze_button:
        if not product_input.strip():
            st.warning("⚠️ 请输入竞品名称或产品描述")
        else:
            # 新增步骤：执行实时搜索
            with st.status("🛸 正在全网搜集情报...", expanded=True) as status:
                st.write("正在检索最新市场动态 (Tavily)...")
                web_context = fetch_competitor_context(product_input)
            
                st.write("情报已汇总，正在进行逻辑建模...")
                analysis_result = perform_analysis(model, product_input, web_context)
            
            if analysis_result:
                status.update(label="✅ 深度分析完成", state="complete", expanded=False)
                add_to_history(product_input, analysis_result)
                # 保存到 session state
                st.session_state['last_analysis'] = analysis_result
                st.session_state['last_product'] = product_input
                markdown_report = generate_markdown_report(product_input, analysis_result)
                st.session_state['last_markdown'] = markdown_report
                st.rerun()  # 重新运行以显示结果
    
# ==================== 显示分析结果 ====================
    if 'last_analysis' in st.session_state and st.session_state.get('last_analysis'):
        analysis_result = st.session_state['last_analysis']
        product_name = st.session_state.get('last_product', '未知产品')
        
        st.success(f"✅ {product_name} 分析已就绪")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔧 技术栈", "🎯 场景适配", "🛡️ 数据护城河", "⚡ 交互痛点", "💰 商业化", "💡 竞争建议"
        ])
        
        # 统一渲染样式
        def display_content(title, key):
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            st.markdown(analysis_result.get(key, "暂无内容"))
            st.markdown('</div>', unsafe_allow_html=True)

        with tab1: display_content("技术栈与模型依赖", "model_stack")
        with tab2: display_content("核心解决的细分场景", "scene_fit")
        with tab3: display_content("数据闭环与护城河", "data_moat")
        with tab4: display_content("交互痛点分析", "ux_friction")
        with tab5: display_content("商业化价值评估", "commercial_roi")
        with tab6:
            st.markdown('<div class="analysis-section" style="background-color: #e3f2fd; border-left: 5px solid #1f77b4;">', unsafe_allow_html=True)
            st.markdown("### 💡 错位竞争建议")
            # 使用你之前的格式化建议函数
            formatted_advice = format_competitive_advantage(analysis_result.get("strategy_advice", ""))
            st.markdown(formatted_advice)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        markdown_report = generate_markdown_report(product_name, analysis_result)
        
        safe_product_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"调研报告_{safe_product_name}_{datetime.now().strftime('%Y%m%d')}.md"
        
        st.download_button(
            label="📥 下载完整分析报告 (Markdown)",
            data=markdown_report,
            file_name=filename,
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )
if __name__ == "__main__":
    main()

