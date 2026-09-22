# 🛢️ LangGraph SQL Agent

基于 **LangGraph** 的中文 Text-to-SQL 智能体：把自然语言问题转成 SQL、执行查询、用自然语言分析结果并自动生成图表。

## 核心工作流

```
用户问题 → 意图识别 → Schema检索 → SQL生成 → SQL检查
              │                                  │
              └────(非数据问题→直接回答)          ▼
                                              执行
                                          ┌─────┴──────┐
                                       失败→修复→重新生成  成功→分析
                                          │                │
                                          └──(重试耗尽→报错)  可视化 → 回答
```

对应 LangGraph 中的 11 个节点：`classify_intent` / `direct_answer` / `retrieve_schema` / `generate_sql` / `validate_sql` / `execute_sql` / `repair_sql` / `analyze` / `decide_chart` / `finalize` / `finalize_error`。

## 特性

- ✅ 完整的 Agent workflow（不是把 prompt 一次性丢给 LLM）
- ✅ SQL 安全校验：只允许单条 `SELECT`，禁止写操作
- ✅ 执行失败自动携带错误信息重试（默认最多 3 次）
- ✅ 自动选择图表类型（柱状 / 折线 / 饼图 / 散点），带启发式兜底
- ✅ 内置 SQLite 电商示例数据，开箱即用
- ✅ Streamlit 交互界面 + 命令行两种用法

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 国内网络慢的话，用清华镜像：
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

默认使用 DeepSeek（OpenAI 兼容接口），也可以换成 OpenAI / 其他兼容服务，修改 `.env` 中的
`LLM_BASE_URL` 和 `LLM_MODEL` 即可。

### 3. 启动

Web 界面：

```bash
streamlit run app.py
```

命令行测试：

```bash
python cli.py "2024 年每月销售额是多少？"
```

## 示例问题

- 2024 年每月销售额是多少？（折线图）
- 各品类的销售额占比？（饼图）
- 哪个城市的客户消费最高？（柱状图）
- 平均客单价最高的前 5 个客户是谁？
- 线上和门店渠道的订单量对比？

## 项目结构

```
data_analyst_agent/
├── app.py              # Streamlit 界面
├── cli.py              # 命令行入口
├── requirements.txt
├── .env.example
├── data/               # SQLite 数据库（自动生成）
└── src/
    ├── config.py       # 配置
    ├── state.py        # Agent 状态定义
    ├── prompts.py      # 各节点提示词
    ├── db.py           # SQLite + 示例数据 + schema 内省
    ├── graph.py        # LangGraph 工作流
    └── viz.py          # 图表生成
```

## 换用你自己的数据

1. 把数据导入 SQLite（或直接替换 `src/db.py` 里的建表 / 灌数逻辑）。
2. 修改 `.env` 里的 `DB_PATH` 指向你的数据库。
3. 在 `src/db.py` 的 `get_relevant_schema` 里补充「关键词 → 表」映射即可。
