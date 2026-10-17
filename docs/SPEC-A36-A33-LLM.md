# A3.6 讨论地图 + A3.3 社区简报 实施规格（LLM 阶段）

> 版本：2026-07-31 ｜ 读者：实施 Agent ｜ 前置文档：`docs/SPEC-ROADMAP.md` v4.1（§A3.3、§A3.6、ADR-10 翻案、ADR-17）
> 本文只覆盖这两个 LLM 阶段管线；其余阶段已完成或不在本次范围。

## 0. 现状（事实，勿重复验证）

- 发布库 `db/content.v2026.07.28.db`（11.35GB）：评论已入库
  `community_document` 1,102,297 条（bgm_ep_comment 945,037 /
  bgm_review 15,456 / bgm_subject_comment 141,804），`source_document`、
  `document_chunk` 配套完整。
- 有评论的单集 15,603 个；≥10 条评论的集 11,854 个。
- 评论量 top200 动画：覆盖 5,308 集、58.1 万条评论，其中 3,926 集达地图门槛。
- embedding 角色已配置（runtime_model_config.json 的 roles.embedding）；
  chat/organizer 用 DeepSeek（key 在 `.env`：`DEEPSEEK_API_KEY`）。
- 可复用设施：`src/mikirin/organize.py`（按批组包、内容寻址缓存、
  失败容错、并发 workers、error.json 隔离）、`src/mikirin/cache.py`、
  `src/mikirin/atomic.py`（atomic_write_json）。
- 硬纪律：Agent/查询运行时纯标准库；LLM 输出全部进内容寻址缓存；
  不修改发布库，产物先落 `data/staging/`，随下次构建入库。

## 1. A3.6 讨论地图

### 1.1 定位修正（重要，已写入规格书）

社区讨论的主体是**分析**（跨集引用、叙事结构、演出/音乐技法、
元解读分歧、改编对照），不是"喜欢/不喜欢"投票。
**做讨论主题发现，不做立场抽取。** 主题不带极性、不数人头。

### 1.2 采样器（确定性，零模型）

对每集（episode_entity_id）：
1. 取该集全部评论（`community_document` join `source_document`）；
2. 去重：文本前 40 字符相同或字符 4-gram Jaccard ≥0.85 视为重复，
   每组保留最长一条；
3. 采样目标 ~30 条：按 (rating 优先、长度 40~600 字符优先、
   时间分散、作者去重 ≤2) 分层选取；
4. 覆盖门槛：去重后 **<10 条不生成**（记录到 skipped 清单）。

### 1.3 LLM 任务（每集一次调用）

system prompt 要点：
- 你是动画评论分析员；只根据给定评论工作，禁止引入外部知识；
- 任务：归纳观众在讨论什么**主题**，不是判断喜欢不喜欢；
- 每个主题输出：一句主题描述（≤30 字）、方面标签（闭集：
  animation/direction/music/story/character/pacing/adaptation/
  voice_acting/other）、2 条代表引文（**必须是所给评论的逐字段落**，
  附上评论编号）；
- 主题 4~8 个；输出 JSON：`{"themes":[{"theme","aspect","quotes":[{"comment_id","quote"}]}]}`。

user payload：`{"subject": "…", "episode_sort": N, "comments": [{"comment_id","content"}]}`

### 1.4 引文逐字硬校验（出库门禁，必须实现）

- 每条 quote 必须是对应 comment_id 评论内容的**逐字子串**
  （忽略首尾空白）；对不上 → 删除该主题，记入 `dropped_themes.jsonl`；
- 全部主题被删的集 → 记 `empty_maps.jsonl`，不算失败；
- **禁止"修补"引文**，宁可整题丢弃。

### 1.5 产物

```
data/staging/discussion_maps/<run_id>/
  maps.jsonl              每行一集：{episode_entity_id, subject_entity_id,
                           episode_sort, themes[...], coverage{comments, distinct_authors},
                           extraction_model, prompt_version}
  dropped_themes.jsonl    引文校验丢弃
  empty_maps.jsonl
  skipped_episodes.jsonl  覆盖门槛未达
  manifest.json           批次/成本/计数
```

`prompt_version = "theme-v1"`；extraction 走 organize.py 同款的
ModelInvocation 缓存（kind='discussion_theme'）。

### 1.6 成本闸门（按此顺序，不得跳步）

1. **试点**：评论量 top 50 动画中达门槛的集（约 1,000 集）；
   完成后计算：引文命中率（目标 ≥98%）、每集主题数分布、空图率（<10%）；
2. **扩量**：top 200（3,926 集），同样指标复核；
3. **全量**：11,854 集，须用户确认后再跑。

并发 6、batch 组包沿用 organize 模式；失败批次落 error.json 不阻塞。

## 2. A3.3 社区简报

### 2.1 范围

每部热作品一次调用。**禁止全库预生成**；本次只跑 top 200（试点可先 20 部）。

### 2.2 证据采样（确定性，硬预算）

每部作品：
- 长评段落 ≤30 段（单段 ≤1,200 字符，按评分/长度选）；
- 全集短评 ≤50 条（≤400 字符，时间分层）；
- 单集评论：评论数 top 5 的集各 ≤10 条；
- 全局：同作者 ≤2、必含低评分样本（1~5 分）若存在；
- estimated tokens ≤24,000，超出按 长评→短评→单集 顺序截断。

### 2.3 LLM 任务

system prompt 要点：
- 只根据所给证据；禁止引入外部知识；
- **禁用"公认/普遍认为/社区共识"等强表述**（v4.1 禁令）；
  允许"抽样证据中多名用户提到…"；
- 输出 JSON：`{"reception_summary", "strengths[], "weaknesses[]",
  "discussion_points[]", "key_episodes[{sort, reason, evidence_comment_ids[]}]",
  "notable_disagreements[]"}`；
- 每个论点附 evidence_comment_ids。

### 2.4 校验与产物

- evidence_comment_ids 必须在输入集合内（程序化校验，对不上删该论点）；
- 产物 `data/staging/community_briefs/<run_id>/briefs.jsonl`，
  每行含 subject_entity_id、coverage（各来源条数/独立作者数/集数范围）、
  source_digest、sampling_policy_version、prompt_version="brief-v1"、
  model_fingerprint。

## 3. 工程约束

- 纯标准库（采样/校验/缓存/并发），不引第三方库；
- 所有 LLM 调用走 `ModelInvocation` 内容寻址缓存——重跑零重复调用；
- 并发 workers ≤8；402/429 等 API 错误重试一次后落 error.json；
- 测试：采样器（去重/配额/门槛）、引文校验（逐字/篡改/跨评论张冠李戴）、
  预算截断，全部纯标准库内存库测试，不联网不调用真实 API；
- 完成后报告：各阶段调用数、缓存命中率、引文命中率、丢弃率、
  估算成本（按 token 计）。

## 4. 明确不做

- 立场极性标注、观点聚类、"公认"措辞（v4.1 禁令）；
- 入库/发布（产物落 staging 即可，入库走后续构建）；
- embedding（已由 A3.2 完成）；
- 简报注入 topic 层（A3.4 的工作，不在本次）。
