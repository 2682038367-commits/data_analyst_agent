# 🛢️ LangGraph SQL Agent

基于 **LangGraph** 的中文 Text-to-SQL 智能体：把自然语言问题转成 SQL、执行查询、用自然语言分析结果并自动生成图表。

核心亮点：**不止检查 SQL 语法，更检查业务语义** —— 重点解决「SQL 能跑通，但业务答案错误」的问题（例如统计「用户数」时该用 `COUNT(DISTINCT user_id)` 却写成了 `COUNT(user_id)`）。

## 核心工作流

~~~
用户问题 → 意图识别 → Schema 检索 → 问题分类与分析计划
                                          ↓
                              计划第一步 → SQL Reviewer → 执行 → Result Validator
                              ↑                         │
                              └──── 异常修复重试 ──────┘
                                                        ↓
                                               Analyzer 判断证据是否足够
                                                        │
                              ┌──── 需要深挖 ───────────┤
                              │                         │ 证据足够
                              ↓                         ↓
                       自动提出下一问题          多轮证据归因总结
                              │                         ↓
                              └→ SQL → 审查 → 执行     可视化 → 回答
                                      （最多 N 轮）
~~~
SQL 审查分三层：

1. **安全校验**（确定性）：只允许单条 `SELECT`，禁止写操作
2. **编译校验**（确定性）：用 `EXPLAIN` 验证语法 / 表名 / 列名，不真正执行
3. **语义审查**（LLM · SQL Reviewer）：检查 COUNT vs COUNT(DISTINCT)、JOIN 数据膨胀、聚合粒度、时间窗口、NULL 处理、分母定义、GROUP BY 遗漏等 9 类问题

执行后还有一道两层 **Result Validator**：先用确定性规则检查空结果、异常行数、比率/留存率范围、负金额、总量与分组加总、重复记录，再由 LLM 结合问题复核结果粒度、时间窗口与业务量级。发现异常时会把结构化报告反馈给 SQL 生成节点自动修复，并保留每次异常 SQL 的回溯历史。

工作流先由 create_analysis_plan 将问题分类并生成有序查询步骤。每轮结果通过质检后，plan_analysis 先判断证据是否足够；不足时让 Analyzer 给出最多 3 个候选下钻问题及 1～5 的相对信息增益，选择分数最高且不重复、字段可用的方向。新证据可使原计划步骤被改选或提前停止。信息增益是模型给出的相对优先级，不是统计学估计；最终结论仍需由实际 SQL 结果支撑。后续查询失败时，用已取得的证据降级总结。

## 特性

- ✅ 默认 GLM 主模型、DeepSeek 运行时故障切换；仅配置一方 Key 时仍可运行
- ✅ 完整的 Agent workflow（不是把 prompt 一次性丢给 LLM）
- ✅ 查询前分析规划：先分类、确定指标口径和拆解路径，再执行第一条 SQL
- ✅ 自主归因分析：从核心指标出发，自动拆指标、提出下一问题并继续查询
- ✅ 多轮证据链：每轮问题、SQL、结果、候选取舍与 Analyzer 决策均可审计
- ✅ 防失控机制：去重追问、拦截缺字段候选、默认最多深挖 3 轮（可配置 0～5），后续失败时用已有证据降级总结
- ✅ SQL 语义审查：执行前发现「COUNT 该去重却没用 DISTINCT」等业务口径错误
- ✅ SQL 安全校验 + 编译校验：只允许单条 `SELECT`，表 / 列 / 语法错误确定性拦截
- ✅ 执行结果质检：成功执行后再次校验结果合理性
- ✅ 结果异常可追溯：保留异常类型、异常值、对应 SQL 和重试轮次
- ✅ 失败自动携带反馈重试（默认最多 3 次，可在 `.env` 调 `MAX_RETRIES`；结果行数阈值可用 RESULT_MAX_ROWS 配置；自动深挖上限可用 MAX_ANALYSIS_DEPTH 配置）
- ✅ 自动选择图表类型（柱状 / 折线 / 饼图 / 散点），带启发式兜底
- ✅ 内置 SQLite 电商示例数据，开箱即用
- ✅ Streamlit 交互界面 + 命令行两种用法

## Evaluation

项目内置 50 条带 gold SQL 的 benchmark，面向仓库自带的 SQLite 电商示例库，覆盖 8 类问题：

| 类别 | 数量 | 重点能力 |
|---|---:|---|
| 基础聚合 | 8 | SUM、COUNT DISTINCT、客单价、利润 |
| 多表 JOIN | 8 | 客户、订单、明细、商品关联与去重 |
| Top N | 6 | 排序、LIMIT、聚合粒度 |
| 窗口函数 | 6 | RANK、ROW_NUMBER、LAG、累计值 |
| 时间对比 | 8 | 月度、季度、同比、环比 |
| 比率与均值 | 6 | 占比、完成率、复购率、加权均价 |
| Cohort / 留存代理 | 4 | 首购、30 天转化、跨年付费 |
| 复杂归因 | 4 | 指标拆解、渠道/品类/城市贡献 |

三项核心指标严格分开：

- Execution Accuracy：候选 SQL 是否安全并能成功执行。
- SQL Semantic Accuracy：候选 SQL 结果是否与 gold SQL 结果集等价，而不只是“能跑”。比较时忽略行顺序、列别名和全局列顺序。
- Answer Accuracy：LLM Judge 根据 gold 结果判断最终自然语言答案是否正确，阈值为 0.8；建议通过 EVAL_JUDGE_MODEL 配置与被测模型不同的 Judge。

先验证题库，不调用模型：

~~~bash
python evals/run_eval.py --validate-only
~~~

分别运行 Baseline、SQL Reviewer 和完整 Agent：

~~~bash
python evals/run_eval.py --mode baseline --output evals/results/baseline.jsonl
python evals/run_eval.py --mode reviewer --output evals/results/reviewer.jsonl
python evals/run_eval.py --mode full --output evals/results/full.jsonl
~~~

Answer Judge 会额外调用模型。只评 SQL 时可增加 --skip-answer-judge；调试时可用 --limit、--category 或 --case-id 缩小范围。

生成对比报告：

~~~bash
python evals/compare.py \
  evals/results/baseline.jsonl \
  evals/results/reviewer.jsonl \
  evals/results/full.jsonl \
  --output evals/results/comparison.md
~~~

报告包含总体和分类准确率、95% Wilson 置信区间、p50/p95 延迟，以及相对 Baseline 的逐题改进、退化数和 exact McNemar 检验。每次运行还会保存模型、温度、重试配置、benchmark SHA-256、Git commit 和工作区状态等 manifest。仓库不预填虚构分数，所有结果均由实际模型运行生成。

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 国内网络慢的话，用清华镜像：
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. 配置 API Key

~~~bash
cp .env.example .env
~~~

默认以 GLM 为主模型，DeepSeek 为自动 fallback。在 .env 中填入：

~~~ini
LLM_PROVIDER=glm
GLM_API_KEY=你的智谱API_Key
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
~~~

也可用 ZAI_API_KEY 代替 GLM_API_KEY。每次模型请求先调用 GLM；请求抛错时自动以相同输入调用 DeepSeek。只配置 GLM Key 时不会自动回退；只配置 DeepSeek Key 时会直接使用 DeepSeek。GLM_MODEL / GLM_BASE_URL 与 DEEPSEEK_MODEL / DEEPSEEK_BASE_URL 可分别覆盖默认值，互不混用。[智谱官方 LangChain 接入说明](https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction)

如需仅使用 DeepSeek 或其他 OpenAI 兼容服务，可设 LLM_PROVIDER=deepseek，并继续使用原有 LLM_BASE_URL、LLM_MODEL 配置。

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
- 2024 年各月销售额有何变化，异常月份为什么变化？（自动拆解订单数、每单件数和成交单价并继续下钻）

## 项目结构

```
data_analyst_agent/
├── app.py              # Streamlit 界面
├── cli.py              # 命令行入口
├── requirements.txt
├── evals/              # 50 题 benchmark、运行器和对比报告工具
├── .env.example
├── data/               # SQLite 数据库（自动生成）
└── src/
    ├── config.py       # 配置
    ├── state.py        # Agent 状态定义
    ├── prompts.py      # 各节点提示词
    ├── analysis_workflow.py # 自主分析循环、证据快照与防重复逻辑
    ├── result_validator.py  # 查询结果确定性合理性检查
    ├── db.py           # SQLite + 示例数据 + schema 内省
    ├── graph.py        # LangGraph 工作流
    └── viz.py          # 图表生成
```

## 换用你自己的数据

1. 把数据导入 SQLite（或直接替换 `src/db.py` 里的建表 / 灌数逻辑）。
2. 修改 `.env` 里的 `DB_PATH` 指向你的数据库。
3. 在 `src/db.py` 的 `get_relevant_schema` 里补充「关键词 → 表」映射即可。
