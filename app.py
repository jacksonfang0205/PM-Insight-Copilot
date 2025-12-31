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
    /* 气泡样式优化 */
    .analysis-section {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 24px; /* 增加内边距 */
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* 更柔和的阴影 */
        font-size: 16px; /* 调整字体大小 */
        line-height: 1.8; /* 核心：增加行间距 */
    }
    /* 让列表项之间也有呼吸感 */
    .analysis-section ul {
        margin-top: 0;
        padding-left: 20px;
    }
    .analysis-section li {
        margin-bottom: 12px; /* 列表项之间的距离 */
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
            "gemini-3-pro-preview",
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
def create_partial_prompt(product_input: str, keys: List[str], web_context: str) -> str:
    """精简指令：强制纯字符串列表输出"""
    descriptions = {
        "overview": "竞品概况 (产品核心定位、目标人群、3项核心业务线)",
        "ux_features": "功能场景 (3个核心功能、典型使用场景、1个硬核交互痛点)",
        "growth_ops": "运营增长 (3个核心增长手段、目前的运营重心、增长杠杆)",
        "tech_stack": "技术栈分析 (模型依赖、RAG架构特点、技术壁垒)",
        "data_metrics": "商业化指标 (3个主要变现点、估算的 ROI、用户活跃度)",
        "strategy_advice": "错位竞争建议 (3条非对称竞争策略、建议的突破方向)"
    }
    
    task_list = "\n".join([f"- KEY: '{k}'，重点: {descriptions[k]}" for k in keys])
    
    return f"""
你是一位顶级 AI 产品专家。请针对竞品 '{product_input}' 进行深度建模。

【参考情报】
{web_context}

**🎯 格式铁律 (Strict Format Rules)：**
1. 输出必须是标准的 JSON 对象。
2. 键名必须严格匹配: {json.dumps(keys)}。
3. **Value 的结构必须是纯字符串列表 (List[str])**。
   - ✅ 正确: ["核心技术: 使用了Transformer", "数据壁垒: 拥有独家数据集"]
   - ❌ 错误: [{{"title": "核心技术", "desc": "..."}}] (严禁使用对象/字典！)
4. 每个维度输出 3 个核心洞察点。
5. 严禁输出 "Executive Summary"。

**分析维度：**
{task_list}
"""
# ==================== 执行分析 ====================
def perform_analysis(model, product_input: str, web_context: str = "") -> Dict:
    """执行分批分析，并确保所有 Tab 都有回显"""
    batches = [
        ["overview", "ux_features", "growth_ops"],
        ["tech_stack", "data_metrics", "strategy_advice"]
    ]
    
    final_result = {}
    
    with st.status("🔍 正在构建产品模型...", expanded=True) as status:
        for i, batch_keys in enumerate(batches):
            status.write(f"正在分析批次 {i+1}/2...")
            prompt = create_partial_prompt(product_input, batch_keys, web_context)
            
            try:
                # 显式重置响应
                response = model.generate_content(prompt)
                batch_json = parse_json_safely(response.text)
                
                # 检查并修复缺失的键
                for k in batch_keys:
                    if k not in batch_json or not batch_json[k]:
                        batch_json[k] = "⚠️ 该维度未能成功生成，请尝试重新运行。"
                
                final_result.update(batch_json)
            except Exception as e:
                for k in batch_keys:
                    final_result[k] = f"❌ 生成错误: {str(e)}"
        
        status.update(label="✅ 分析完成", state="complete", expanded=False)
    
    # 统一清理文本
    for k in final_result:
        final_result[k] = clean_text(final_result[k])
            
    return final_result


# 辅助函数：安全解析 JSON
def parse_json_safely(text: str) -> Dict:
    """使用正则提取 JSON，应对各种返回格式"""
    try:
        # 先尝试直接解析
        return json.loads(text.strip())
    except:
        try:
            # 如果直接解析失败，寻找第一个 { 和最后一个 }
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            st.error(f"解析 JSON 出错: {e}")
    return {}

def clean_text(text: str) -> str:
    """清理文本，确保 Markdown 在 Streamlit 中完美渲染"""
    if not text or not isinstance(text, str):
        return str(text)
    
    # 修复 JSON 转义导致的换行符失效问题
    text = text.replace('\\n', '\n')
    text = text.replace('\\"', '"')
    
    # 确保标题前有换行，防止 Markdown 粘连
    text = re.sub(r'([^\n])###', r'\1\n\n###', text)
    return text.strip()
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
    
    # 统一 Key 名
    required_keys = ["overview", "ux_features", "growth_ops", "tech_stack", "data_metrics", "strategy_advice"]
    
    if not json_text or json_text == '{':
        return '{\n    ' + ',\n    '.join([f'"{key}": "内容被截断"' for key in required_keys]) + '\n}'
    
    # ... (中间的逻辑保持不变) ...

    # 检查必需字段
    missing_keys = [key for key in required_keys if f'"{key}"' not in json_text]
    
    if missing_keys:
        json_text = json_text.rstrip().rstrip('}').rstrip(',').rstrip()
        json_text += ',\n' if '"' in json_text else ""
        
        for i, key in enumerate(missing_keys):
            json_text += f'    "{key}": "内容被截断"'
            if i < len(missing_keys) - 1:
                json_text += ',\n'
            else:
                json_text += '\n'
        json_text += '}'
    
    if not json_text.rstrip().endswith('}'):
        json_text = json_text.rstrip().rstrip(',')
        json_text += '\n}'
    return json_text




def display_content(title, content, is_strategy=False):
    """
    UI 最终修复版 v6.0 (针对性修复技术栈显示乱码问题)
    1. 智能展平: 如果遇到 {'title':..., 'desc':...} 格式，自动转为 "**Title**: Desc"
    2. 智能解析: 字符串列表还原
    3. 样式优化: 气泡 + 行间距
    """
    st.markdown(f"## {title}")
    
    # 1. 智能解析：如果内容是长得像列表的字符串，强制转回列表
    if isinstance(content, str):
        content = content.strip()
        if content.startswith("[") and content.endswith("]"):
            try:
                content = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                pass

    # 定义配色
    bg_color = "#e3f2fd" if is_strategy else "#ffffff"
    border_color = "#1f77b4" if is_strategy else "#e6e9ef"
    border_left = "8px solid #1f77b4" if is_strategy else f"1px solid {border_color}"
    
    html_inner = ""
    
    if isinstance(content, list):
        html_inner += '<ul style="margin: 0; padding-left: 20px;">'
        for item in content:
            # --- 新增核心逻辑：处理字典类型的 Item ---
            if isinstance(item, dict):
                # 提取字典里的所有值，尝试拼凑成 "标题: 内容" 的格式
                values = list(item.values())
                if len(values) >= 2:
                    # 假设第一个是标题，第二个是描述
                    item_str = f"**{values[0]}**: {values[1]}"
                elif len(values) == 1:
                    item_str = str(values[0])
                else:
                    item_str = str(item) # 兜底
            else:
                item_str = str(item)
            # -------------------------------------

            # 去掉可能的 Markdown ** 符号 (因为我们后面会自己加)
            item_str = item_str.replace("**", "")
            
            formatted_item = item_str
            
            # 逻辑：查找冒号，加粗前半部分
            # 兼容英文冒号(:)和中文冒号(：)
            if ": " in item_str:
                parts = item_str.split(": ", 1)
                formatted_item = f"<strong>{parts[0]}</strong>: {parts[1]}"
            elif "：" in item_str:
                parts = item_str.split("：", 1)
                formatted_item = f"<strong>{parts[0]}</strong>：{parts[1]}"
            
            html_inner += f'<li style="margin-bottom: 16px; line-height: 1.8; color: #333;">{formatted_item}</li>'
        html_inner += '</ul>'
        
    elif isinstance(content, dict):
        html_inner = f"<pre>{json.dumps(content, indent=2, ensure_ascii=False)}</pre>"
    else:
        clean_content = str(content).replace("**", "")
        html_inner = f'<p style="line-height: 1.8; color: #333; margin: 0;">{clean_content}</p>'

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: {border_left};
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        ">
            {html_inner}
        </div>
        """,
        unsafe_allow_html=True
    )

    
def format_bullet_point(text):
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
    return f"""# 竞品分析报告：{product_name}
**生成时间：** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
## 1. 📊 竞品概况
{analysis_result.get("overview", "暂无数据")}

## 2. 🎯 功能与体验分析
{analysis_result.get("ux_features", "暂无数据")}

## 3. 📈 运营与增长策略
{analysis_result.get("growth_ops", "暂无数据")}

## 4. 🏗️ 技术栈分析
{analysis_result.get("tech_stack", "暂无数据")}

## 5. 💰 数据与商业化
{analysis_result.get("data_metrics", "暂无数据")}

## 💡 策略启发与错位竞争
{format_competitive_advantage(analysis_result.get("strategy_advice", ""))}

---
*本报告由 PM Insight Copilot 自动生成*
"""
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
            # 1. 真正的联网搜索步骤
            with st.status("🛸 正在全网搜集情报...", expanded=True) as status:
                st.write("正在检索最新市场动态 (Tavily)...")
                
                # --- 新增代码：实际调用搜索函数 ---
                web_info = fetch_competitor_context(product_input)
                # --------------------------------
                
                st.write("情报已汇总，正在进行逻辑建模...")
                
                # --- 修改调用：把搜索到的 web_info 传进去 ---
                analysis_result = perform_analysis(model, product_input, web_info)
                # ----------------------------------------
            
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
# ==================== 显示分析结果 (竖向长页面模式) ====================
    if 'last_analysis' in st.session_state and st.session_state.get('last_analysis'):
        res = st.session_state['last_analysis']
        product_name = st.session_state.get('last_product', '未知产品')
        
        st.success(f"✅ {product_name} 深度调研报告已生成")
        
        # --- 快捷导航 (可选，放在页面顶部) ---
        st.info("💡 报告已按维度垂直展开，可直接滚动阅读或下载完整报告。")
        
        # --- 定义渲染顺序和标题 ---
        sections = [
            ("🔍 竞品概况", "overview"),
            ("🎯 功能场景分析", "ux_features"),
            ("📈 增长与运营策略", "growth_ops"),
            ("🏗️ 技术栈与底层架构", "tech_stack"),
            ("💰 商业化与价值评估", "data_metrics"),
            ("💡 错位竞争建议", "strategy_advice")
        ]
        
        # --- 循环渲染所有章节 ---
        for title, key in sections:
            is_strategy = (key == "strategy_advice")
            content = res.get(key, "暂无内容")
            
            display_content(title, content, is_strategy=is_strategy)
            
            # 章节间加一个淡淡的分隔线
            if not is_strategy: # 最后一项下面不需要分割线
                st.markdown("---")
            

        # --- 下载按钮 ---
        st.download_button(
            label="📥 下载完整分析报告 (Markdown)",
            data=st.session_state.get('last_markdown', ''),
            file_name=f"调研报告_{product_name}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

