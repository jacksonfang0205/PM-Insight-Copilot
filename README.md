# PM Insight Copilot

AI 驱动的竞品深度分析工具，专为产品经理设计。

## 功能特点

- 🔍 **深度分析**：基于 Google Gemini API 进行智能分析
- 📊 **5 维框架**：Model Stack、Scene-Fit、Data Moat、UX Friction、Commercial ROI
- 💡 **竞争建议**：自动生成错位竞争策略
- 🎨 **专业 UI**：简洁美观的 Streamlit 界面

## 安装步骤

1. **克隆或下载项目**

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置 API Key**（任选一种方式）
   
   **方式 1：使用 .env 文件（推荐本地开发）**
   - 复制 `.env.example` 文件为 `.env`
   - 在 `.env` 文件中填入您的 API Key：
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   
   **方式 2：使用环境变量**
   - 直接设置环境变量：
     ```bash
     export GEMINI_API_KEY=your_api_key_here
     ```
   
   **方式 3：Streamlit Cloud 部署**
   - 在 Streamlit Cloud 的 Settings → Secrets 中添加：
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   
   - 获取 API Key：访问 [Google AI Studio](https://makersuite.google.com/app/apikey)

4. **运行应用**
```bash
streamlit run app.py
```

## 使用方法

1. 在输入框中输入竞品名称或产品描述（例如：ChatGPT、Notion AI、Midjourney）
2. 点击"开始深度分析"按钮
3. 等待 AI 分析完成
4. 查看 5 个维度的详细分析结果
5. 查看错位竞争建议

## 分析框架说明

### 1. Model Stack（技术栈与模型依赖）
分析产品的核心技术栈、AI 模型依赖、技术架构等。

### 2. Scene-Fit（核心解决的细分场景）
明确产品针对的具体使用场景和场景覆盖情况。

### 3. Data Moat（数据闭环与护城河）
分析数据获取、数据质量、数据闭环机制和护城河强度。

### 4. UX Friction（交互痛点分析）
识别用户痛点、交互摩擦点和需要改进的环节。

### 5. Commercial ROI（商业化价值评估）
评估商业模式、定价策略、付费意愿和商业化可持续性。

## 注意事项

- 需要有效的 Google Gemini API Key
- 确保网络连接正常（需要访问 Google API）
- API 调用可能产生费用，请注意使用量

## 许可证

MIT License

