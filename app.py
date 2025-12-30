import streamlit as st
import google.generativeai as genai
from typing import Dict, List
import json
from datetime import datetime
import ast
import re
import os
from dotenv import load_dotenv

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

# ==================== 分析提示词模板 ====================
def create_analysis_prompt(product_input: str) -> str:
    """创建分析提示词"""
    prompt = f"""
你是一位资深的产品经理和竞品分析专家。请对以下竞品或产品进行深度分析：

**分析对象：** {product_input}

**请严格按照以下 5 个维度进行结构化分析，每个维度都需要详细、专业的分析：**

## 1. Model Stack（技术栈与模型依赖）
- 分析该产品使用的核心技术栈
- 识别其依赖的 AI 模型或技术框架
- 评估技术架构的先进性和可扩展性
- 指出潜在的技术风险或依赖

## 2. Scene-Fit（核心解决的细分场景）
- 明确该产品针对的具体使用场景
- 分析场景的细分程度和精准度
- 评估场景覆盖的完整性和深度
- 识别未被充分满足的场景需求

## 3. Data Moat（数据闭环与护城河）
- 分析产品的数据获取渠道和方式
- 评估数据质量和数据量级
- 识别数据闭环的形成机制
- 评估数据护城河的强度和可持续性

## 4. UX Friction（交互痛点分析）
- 识别用户在使用过程中的主要痛点
- 分析交互流程中的摩擦点
- 评估用户体验的流畅度和易用性
- 指出需要改进的交互环节

## 5. Commercial ROI（商业化价值评估）
- 分析产品的商业模式和盈利点
- 评估市场定价策略的合理性
- 分析目标用户群体的付费意愿
- 评估商业化的可持续性和增长潜力

## 6. 错位竞争建议
- 基于以上分析，提供 1-2 条具体的错位竞争策略建议
- 建议应该具有可执行性和差异化优势

**输出格式要求：**
你必须返回一个 JSON 对象，且必须严格包含以下 6 个字段（Key 必须完全一致，不能多也不能少）：
{{
    "model_stack": "详细分析内容...",
    "scene_fit": "详细分析内容...",
    "data_moat": "详细分析内容...",
    "ux_friction": "详细分析内容...",
    "commercial_roi": "详细分析内容...",
    "strategy_advice": "错位竞争建议内容..."
}}

重要要求：
1. 必须返回有效的 JSON 格式，且 JSON 必须完整（不能截断）
2. Key 名称必须完全匹配上述 6 个字段名
3. 每个字段的内容必须完整，不能截断
4. 如果内容较长，请适当精简，确保 JSON 结构完整
5. 请确保分析深入、专业，并基于实际的产品理解
6. 特别注意：JSON 字符串中的引号必须正确转义，确保 JSON 格式有效
"""
    return prompt

# ==================== 执行分析 ====================
def perform_analysis(model, product_input: str) -> Dict:
    """执行竞品分析"""
    prompt = create_analysis_prompt(product_input)
    
    try:
        with st.spinner("🔍 正在进行深度分析，请稍候..."):
            response = model.generate_content(prompt)
            response_text = response.text
        
        # 直接解析 JSON 响应（因为已配置 response_mime_type="application/json"）
        try:
            # 清理可能的代码块标记
            json_text = response_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:].strip()
            if json_text.startswith("```"):
                json_text = json_text[3:].strip()
            if json_text.endswith("```"):
                json_text = json_text[:-3].strip()
            
            # 解析 JSON
            analysis_result = json.loads(json_text)
            
            # 验证必需的字段
            required_keys = ["model_stack", "scene_fit", "data_moat", "ux_friction", "commercial_roi", "strategy_advice"]
            missing_keys = [key for key in required_keys if key not in analysis_result]
            
            if missing_keys:
                st.warning(f"⚠️ 响应缺少以下字段: {', '.join(missing_keys)}，将使用默认值填充")
                for key in missing_keys:
                    analysis_result[key] = "暂无数据"
            
            # 清理所有字段中的转义字符
            for key in analysis_result:
                if isinstance(analysis_result[key], str):
                    analysis_result[key] = clean_text(analysis_result[key])
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            # JSON 解析失败，尝试修复截断的 JSON
            try:
                # 尝试修复未闭合的字符串
                json_text_fixed = fix_truncated_json(json_text)
                analysis_result = json.loads(json_text_fixed)
                
                # 验证必需的字段
                required_keys = ["model_stack", "scene_fit", "data_moat", "ux_friction", "commercial_roi", "strategy_advice"]
                missing_keys = [key for key in required_keys if key not in analysis_result]
                
                if missing_keys:
                    st.warning(f"⚠️ JSON 被截断，缺少以下字段: {', '.join(missing_keys)}，将使用默认值填充")
                    for key in missing_keys:
                        analysis_result[key] = "内容被截断，请重试分析"
                
                # 清理所有字段中的转义字符
                for key in analysis_result:
                    if isinstance(analysis_result[key], str):
                        analysis_result[key] = clean_text(analysis_result[key])
                
                st.warning("⚠️ JSON 响应被截断，已尝试修复。建议重试以获得完整结果。")
                return analysis_result
                
            except (json.JSONDecodeError, Exception) as fix_error:
                # 修复失败，显示详细错误信息
                st.error(f"❌ JSON 解析失败: {str(e)}")
                st.error("**可能的原因：**")
                st.write("- JSON 响应被截断（内容过长）")
                st.write("- JSON 格式错误（引号未正确转义）")
                st.error("**响应内容（前 1000 字符）：**")
                st.code(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
                st.warning("⚠️ 请重试分析，或尝试简化产品描述")
                return None
    
    except Exception as e:
        st.error(f"❌ 分析过程中出现错误: {str(e)}")
        return None

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
    
    # 处理转义字符，按顺序处理以避免重复替换
    # 先处理双反斜杠的情况（如果原本就是转义的）
    text = text.replace('\\\\n', '\n')
    text = text.replace('\\\\t', '\t')
    text = text.replace('\\\\r', '\r')
    
    # 然后处理单反斜杠的转义字符
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    text = text.replace('\\r', '\r')
    
    # 清理多余的空行（连续3个或更多换行符替换为2个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
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
            # 执行分析
            analysis_result = perform_analysis(model, product_input)
            
            if analysis_result:
                # 添加到历史记录（包含分析结果）
                add_to_history(product_input, analysis_result)
                
                # 保存到 session state
                st.session_state['last_analysis'] = analysis_result
                st.session_state['last_product'] = product_input
                markdown_report = generate_markdown_report(product_input, analysis_result)
                st.session_state['last_markdown'] = markdown_report
                st.rerun()  # 重新运行以显示结果
    
    # 显示分析结果（从 session_state 读取，确保下载后不消失）
    if 'last_analysis' in st.session_state and st.session_state.get('last_analysis'):
        analysis_result = st.session_state['last_analysis']
        product_name = st.session_state.get('last_product', '未知产品')
        
        # 显示分析结果
        st.success("✅ 分析完成！")
        st.markdown("---")
        
        # 使用 Tabs 展示 6 个维度（包括错位竞争建议）
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔧 技术栈",
            "🎯 场景适配",
            "🛡️ 数据护城河",
            "⚡ 交互痛点",
            "💰 商业化",
            "💡 竞争建议"
        ])
        
        with tab1:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("### 技术栈与模型依赖")
            content = clean_text(analysis_result.get("model_stack", "暂无数据"))
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("### 核心解决的细分场景")
            content = clean_text(analysis_result.get("scene_fit", "暂无数据"))
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("### 数据闭环与护城河")
            content = clean_text(analysis_result.get("data_moat", "暂无数据"))
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("### 交互痛点分析")
            content = clean_text(analysis_result.get("ux_friction", "暂无数据"))
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("### 商业化价值评估")
            content = clean_text(analysis_result.get("commercial_roi", "暂无数据"))
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab6:
            strategy_advice = analysis_result.get("strategy_advice", "")
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("### 错位竞争建议")
            if strategy_advice:
                formatted_advice = format_competitive_advantage(strategy_advice)
                # 再次清理格式化后的内容
                formatted_advice = clean_text(formatted_advice)
                st.markdown(formatted_advice)
            else:
                st.markdown("暂无数据")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 导出 Markdown 报告（始终显示，即使点击下载也不会消失）
        st.markdown("---")
        markdown_report = st.session_state.get('last_markdown', generate_markdown_report(product_name, analysis_result))
        
        # 生成文件名
        safe_product_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"竞品分析_{safe_product_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        st.download_button(
            label="📥 下载 Markdown 报告",
            data=markdown_report,
            file_name=filename,
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

