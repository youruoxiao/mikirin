# Mikirin 实施路线图规格书（A2–A5 / B2–B3）

> 版本：2026-07-29 v4.1（实施冻结版：弱锚点/共识判定/向量索引一致性边界固化）｜ 读者：实施 Agent/工程师 ｜ 前置：无（本文自包含）
>
> 阶段：**A2 实体合并构建**（全细节）→ **A3 社区文档层** →
> **A4 黑话发现管线 ‖ B2 记忆管线+人格**（并行）→ **A5 图派生** → **B3 主动搭话**。
> 所有标注【ADR-n】的决策见 §10 架构决策记录，**不得在未翻案 ADR 的情况下偏离**。

---

## 0. 项目背景与现状（事实，勿重复验证）

Mikirin 是本地优先的动画领域知识库，位于 `D:\code\mikirin`，Python 3.12+ 单进程。
Agent/查询运行时**纯标准库**；离线管线允许第三方库【ADR-5】。
设计基线 v1.2：稳定实体 ID（绝不按同名合并）、raw/staging/published 分层、
content.db 为可重建构建产物、LLM 输出全部进内容寻址缓存。

当前发布 `db/content.v2026.07.21.db`（约 6.59GB，`db/current.json` 指向）：
BGM Archive 全量（658,248 条目 / 216,126 角色 / 332 万别名）+
萌百 45,999 角色（未解析实体 `uuid5:character:*`，事实在 `entity_fact`）+
作画 Wiki 2,001 篇文档 + FTS5 trigram。

实体消歧已完成，60,121 对候选全部有终态，**22,027 对 auto_same**：

| 决策产物 | 位置 |
|---|---|
| 合并汇总 | `data/staging/resolution_rules/rules-20260727/resolution-compiled.jsonl` |
| **人工裁决（最高优先级）** | `data/manual/entity_resolution.jsonl`（78 条：same 12 / different 66） |
| 规则层裁决 | 同目录 `rule-decisions.jsonl` |
| LLM flash/pro 裁决 | `data/staging/deepseek_resolution/person-v4f-20260728/`、`pro-review-20260728/` |

---

## 1. A2 实体合并构建（全细节）

### 1.1 目标与原则

22,027 对 auto_same 应用为**实体合并**【ADR-1】：萌百实体并入 BGM 权威实体
（`bgm:character:*` 为规范侧），旧 ID 经 `entity_redirect` 永久可解析。
合并是构建期重放，不是原地 UPDATE；新库旁路构建，旧库不动。

### 1.2 合并映射构造（优先级：人工 > pro > flash > 规则）

输入：`resolution-compiled.jsonl`（final_decision/decision_source/confidence）+
`data/manual/entity_resolution.jsonl`（decision/decided_by）。

规则：
1. `final_decision=auto_same` 进入合并映射；
2. 人工 `decision=same` **强制加入**；人工 `decision=different` **强制否决**（纠错通道）；
3. 冲突检测（命中即拒绝发布并输出清单）：
   - 同一萌百实体合并到 ≥2 个不同 BGM 实体；
   - redirect 成链（A→B 且 B→C，必须归一到 C）；
   - 同一 BGM 实体接收 ≥2 个萌百实体且决策全部非 human（列出供知情，不必然拒绝）。
4. 复核尾巴处置【ADR-7】：v4p 高置信 different 全部接受为否决对（候选 status='rejected'）；
   低置信/被标记的 same **一律不合并**；uncertain 保持 'review' 挂账。
5. 萌百限定角色（无 BGM 对应物，约 1.9 万）**保留为独立实体**【ADR-8】。

### 1.3 合并执行（构建库上，每对幂等）

对每对 old=萌百实体 / canonical=BGM 实体，事务内执行：

1. `entity_alias.entity_id`：old → canonical；同 (entity_id, normalized_alias, source_kind)
   冲突保留置信度高者，被替换行记录 `rejected/merge-alias-conflicts.jsonl`；
2. `entity_fact.entity_id`：old → canonical。**两边事实全留**【ADR-2】——
   UNIQUE(entity_id, fact_kind, source_asset_id) 天然共存，不做赢家删除；
   查询展示层 BGM 来源优先；互相矛盾的事实对（如性别打架）输出到
   `data/manifests/quality/fact-conflicts-<build_id>.jsonl` 供日后人工抽查；
3. `source_entity`：萌百侧 `entity_id` 改为 canonical，`mapping_status='resolved'`；
4. 插入 `entity_redirect(old → canonical, reason='auto_same:<source>', decided_at)`；
5. `resolution_decision` 每对一条（schema 以 `src/mikirin/schema.py` 为准，
   candidate 关联放 evidence_json）；
6. `resolution_candidate.status`：合并对置 'merged'，否决对置 'rejected'，
   丢弃置 'dropped'，其余 'review'；
7. `entity` 表 old 实体保留（redirect 锚点），`status='merged'`。

### 1.4 查询侧契约

- `search_entities` / `search_anime_context` 命中被合并实体时跟随 redirect 到规范实体；
- `entity_fts` 在合并执行**之后**重建（搜"加特"命中 BGM 角色且带萌百数据）；
- 旧 UUID5 ID 直查返回规范实体并附 `redirected_from` 字段。

### 1.5 门禁与黄金集【ADR-3】

- **硬门禁：78 条人工裁决**——same 必须存在正确 redirect，different 两实体间
  不得存在 redirect，任一不过不发布；
- **监控指标（不阻断）**：与 500 条 LLM 标注的一致率，>2% 不一致输出清单；
- DQ：redirect 数 == 最终 `merge_map` 计算数（±0，不硬编码 22,027）；
  规范实体携带萌百来源别名/事实的占比 ≥ 99%；
  **active 实体减少数 == redirect 数**（旧实体保留但 status='merged'，
  entity 总行数不变——门禁按 status 计数，不按总行数）。

### 1.6 实施步骤

新增 `src/mikirin/merge_entities.py`：
`load_merge_map`（构造+校验，§1.2）/ `apply_merges`（§1.3，幂等）/
`verify_golden`（§1.5）。
CLI：`build-bgm-content` / `finalize-content` 增加 `--resolution-compiled` /
`--manual-resolution`（默认自动发现 §0 表格中的路径）；
装载完成后、FTS 重建前执行 apply_merges。

测试（纯标准库，`tests/test_merge_entities.py`）：迷你内存库全字段验证、
幂等重跑、一人并两物拒绝、人工否决优先、redirect 链归一、查询跟随 redirect。

验收命令：
```powershell
mikirin finalize-content --version <新版> --resolution-compiled ... --manual-resolution ...
mikirin search "加特"      # 命中 bgm:character:13867，带萌百萌点/别名
mikirin search "泉此方"    # 事实同时含 BGM 与萌百来源
```

### 1.7 错合并拆分契约【ADR-4】

不做专用工具。拆分时向 `data/manual/entity_resolution.jsonl` 追加：

```json
{"operation": "split", "from_entity": "<canonical_entity_id>",
 "assignments": {"<source_entity_id>": "<new_or_old_entity_id>"},
 "reason": "...", "decided_by": "human", "decided_at": "..."}
```

下一次构建重放：按 assignments 把 source_entity/alias/fact 重新归属，
redirect 失效化。工具等第一个真实错合并出现再做。

---

## 2. A3 社区文档层（A3.1~A3.5）

依赖：BGM 爬虫完成（42,946 页，已于 2026-07-28 完成）。
**剧透作用域标注已取消【ADR-9】**——软纪律见 §5.4。

### A3.1 社区文档确定性入库【ADR-17：本阶段零 LLM 零 embedding】

入库命令（现有）：

```powershell
mikirin stage-bgm-web 20260728 --require-complete
mikirin build-bgm-content 2026-07-21 <asset_manifest> --moegirl-version ... `
  --atwiki-version ... --bgm-web-version 20260728 --version <新版>
```

文档关系补强：
- `community_document.doc_kind`：ep_comment（作品+单集双锚）/
  subject_comment（仅作品锚，**绝不伪造单集关系**）/ subject_review（仅作品锚）；
- `parent_document_id` / `root_document_id` 保留评论线程，
  顶层评论与回复各自独立成文档，不拼接；
- `document_section`：长评按 HTML 结构切分（h1/h2/h3/blockquote/p），
  切块边界：标题 > 小标题 > 段落 > 句子 > 长度兜底（600~1200 中文字符）；
- `document_anchor`：通用锚点层（强关系 primary_* 来自页面结构）。
  **别名命中 ≠ 实体解析成功**【v4.1-1】：AC 命中后取该别名的全部候选
  实体，在当前 subject 的角色/staff 作用域内消歧——唯一候选才写
  `resolution_status='resolved'`，多候选写 `'ambiguous'` 并保留
  candidate_json（或按配置跳过）；单字及高歧义别名默认不形成
  resolved anchor，留给 A4 高门槛机制；
- `document_episode_mention`：正文推断集数的弱关系【v4.1-2】。
  必须先构建**作品内章节引用表**（subject_id + episode_type +
  display_number + sort + 别名 → episode_id），数字集数不得直接等于
  episode sort（存在 SP/OVA/总集篇/小数 sort）；"上一集/下一集"仅在
  文档自带强 episode 锚（ep_comment）时解析，长评中的相对表达放弃；
  "最终话"仅在作品已完结且正篇末集唯一时映射；"前三集"等范围表达
  保存为 episode range，不伪装成单集关系。推不出就不抽，**宁缺勿错**；
- `document_chunk.context_header` 确定性生成（作品名/文档类型/长评标题/
  章节标题/已识别实体）；FTS 索引 `search_text = context_header + content`，
  展示引用仍用 `content_text`。

### A3.2 单集卡与社区切片语义索引【ADR-10 翻案】

**embedding 角色配置**（provider 无关，不复用 chat provider）：

```json
{"role": "embedding", "provider": "openai-compatible", "base_url": "...",
 "model": "...", "dimensions": 1024, "distance": "cosine",
 "query_prefix": "", "document_prefix": "", "normalize": true}
```

向量兼容拆分为两级指纹：
- `embedding_fingerprint = provider + canonical_base_url + model +
  returned_dimensions + normalize + query_prefix + document_prefix +
  input_normalization_version`；
- `index_fingerprint = embedding_fingerprint + projection_hash +
  signature_bits + quantization_kind/version + card_schema_version +
  chunk_schema_version`。
换任一指纹必须旁路重建索引，禁止混用。**运行时 embedding 配置与已发布
索引不匹配时，必须禁用向量召回并明确降级到 FTS**——禁止拿新模型查询
向量去搜旧索引。密钥仅存 `.env`。BGE-M3 类多语言模型适合中日跨语言
召回，但第一版只用 dense vector，不引入 sparse/多向量模式。

**向量化范围（严格限定，不扩散到实体层）**：
1. 有有效内容单集的 `episode_retrieval_card`；
2. 社区长评/单集评论/全集短评的检索切片；
3. A3.3 生成的作品社区简报。

**单集卡确定性生成**（LLM 摘要只是可选增强，不是前置条件）：

```
episode_retrieval_card =
  作品名及别名 + 集数/标题 + 官方剧情简介
+ 能从确定性来源确认的本集角色及 staff（禁止为补齐字段引入 LLM）
+ 评论高区分度 n-gram + 去重后少量代表性评论
```

**本地两级向量索引**（运行时零第三方依赖）：
- 离线管线可用 NumPy 生成随机投影矩阵；
- 签名存**固定宽度二进制文件 + mmap**（不要用 SQLite 逐行扫描）；
  SQLite 只保存 `vector_row_id → 文档/章节` 元数据；
- 粗排：Python 大整数 `xor + bit_count()` 全量扫签名取 Top 300~500；
  精排：只用**归一化 int8** 一种格式（查询向量对称量化），
  纯 Python 点积取 Top 20~50；
- 不引入 FAISS/hnswlib/sqlite-vss/远程向量库；
- **性能门禁（按真实 chunk 数验收，不按文档页数承诺）**：
  索引 ≤250,000 条用全量签名扫描；超限或 p95 超标时升级为
  SimHash 分带 LSH 候选桶（仍纯标准库），升第三方 ANN 必须另立 ADR；
  初值目标：episode_card 全局召回 p95 ≤150ms、社区 chunk 全局 p95 ≤500ms、
  作用域内 p95 ≤150ms、粗排候选 Recall@500 ≥0.95。

**2026-08-02 实测决策**：在 1,113,721 个真实社区 chunk 上，原始
8×16-bit/radius-2 LSH 的候选 Recall@500 未达门禁；扩大候选桶达到门禁时
已接近全库扫描。改用 `heapq.nsmallest(500)` 流式全签名扫描后，全局 p95
为 309.6ms、作用域 p95 为 69.1ms、候选 Recall@500=1.0，均通过上述门禁，
因此当前发布索引采用全扫；LSH 实现保留为将来数据规模或 p95 再次超标时的
可选路径。该决策不改变 mmap 签名 + int8 精排与运行时零第三方依赖约束。

### A3.3 作品社区简报（热预生成、冷按需）

**简报是 A3 的检索产物，不是全库无差别预生成任务。**

生成流程（成本控制的核心在确定性选证据，不在模型）：

```
全部社区文本 → 确定性统计分桶 → 去重/作者限额/时间分层/观点多样性采样
→ 有限证据包（长评段 ≤30、短评 ≤50、单集评论每集限额总 ≤50、
  同作者 ≤2、必含正/负/争议、总输入硬封顶）
→ 一次 LLM 生成 community_report
```

- 热作品（用户在看/收藏/当季新番/高频，预计 ≤200 部）构建期预生成；
  **热作品清单必须作为版本化构建输入落盘**（生成规则 + 用户状态快照
  摘要），禁止构建过程隐式读取实时查询历史而产生不可重放的 content.db；
- 冷作品首次进入话题时按需生成：**写入内容寻址 llm_cache 或运行时
  overlay，不修改当前发布的 content.db 与向量索引**；当前会话直接
  注入 topic 层（此时本来就在聊该作品，无需向量化）；下次旁路构建
  时才物化与向量化；
- 报告带 coverage 元数据（各来源条数、独立作者数、集数范围）；
- **第一版禁止使用"公认/普遍认为/社区共识"等强表述**——没有观点归并
  机制时，COUNT(DISTINCT author_hash) 只能证明多人发言，不能证明多人
  持同一立场。只允许弱归纳（"抽样证据中多名用户提到…"）与确定性
  指标（评论量峰值、评分分布、互动数据、完全相同的关键词模式）；
  强共识表述留给未来观点归并机制支撑后再开放；
- 正负/争议分桶是**启发式采样桶，不是情感结论**：仅在评分元数据或
  规则词典足够可靠时分桶，否则按来源/作者/时间/词面多样性采样，
  由最终回答模型识别分歧，不得假设所有证据立场一致；
- **输入预算硬编码**：estimated_max_tokens ≤24,000；长评段单条 ≤1,200
  字符、评论单条 ≤400 字符；确定性采样必须在调用 LLM 前完成，超预算
  按来源配额/作者多样性/信息增益截断，不依赖模型自动截断；
- 缓存键绑定 source_digest + sampling_policy_version + prompt_version +
  model_fingerprint，任一变化产生新键。

### A3.4 话题锁定、简报注入与对话焦点

提示词在 stable/context 与 volatile 之间插入 **topic 层**：

```
stable → context → topic（community_report + 作品卡）→ volatile → conversation
```

- topic 层在同一作品话题内**字节完全一致**（时间戳/排序/空白不变），
  吃前缀缓存；仅在话题切换、用户要求刷新、或发布库切换且报告版本
  变化时替换；无锁定作品时使用**固定空围栏**；对比讨论最多同时加载
  两个作品简报；
- 运行时维护轻量对话焦点
  `{active_subject_id, active_episode_id, active_entities, resolved_at_turn, confidence}`：
  "那集的分镜谁画的"这类指代直接解析焦点 + SQL 查询，不重新语义检索；
  **作品切换时清空 active_episode_id；焦点超过 N 轮（初值 10）未被
  引用则置信度衰减，不得永久黏住。**

### A3.5 混合检索与盲评

**路由优先级**（embedding 是无锚查询的兜底主力，不是所有查询的第一步）：

```
明确实体/集数 → 对话焦点 → 别名与 FTS → 向量语义召回 → LLM 候选消歧
```

**晚聚合**（"那场 live 翻车的是哪集"的双路召回）：

```
A. 单集卡向量召回
B. 单集评论 chunk 向量召回 → 按 episode_id 聚合
episode_score = 0.50×episode_card + 0.30×best_chunk
              + 0.10×top3_chunk_mean + 0.10×lexical
```

"哪个番"问题再按 chunk → episode → subject 两级聚合，
展示 候选作品 → 候选章节 → 匹配描述 → 支撑原文。

证据包组装（沿用 v3 规则）：official-facts / scope-summary /
supporting / dissenting（评价类必带反方）/ deep-analysis；
同作者 ≤2 条；单篇长评 ≤40%；每条带 url 与 author_hash。

**验收**：60 题对照集（单集事实/单集评价/整体口碑/深度解读/跨集变化/
争议观点/黑话/"哪一集"/"哪个番"），对比 扁平 FTS / 锚点+父子 /
+证据包+简报+向量；指标 scope_accuracy、evidence_recall@8、
citation_precision、contradiction_coverage、author_diversity；
盲评裁判：用户本人【ADR-6】。

---

### A3.6 讨论地图层（社区"在聊什么"的可审计索引）

动机（实测孤独摇滚 ep8 全部高赞评论后修正）：社区讨论的主体不是
"喜欢/不喜欢"的投票，而是**分析**——跨集引用、叙事结构辩论、
演出/音乐技法拆解、元解读分歧、改编对照。因此不做立场抽取，
做**讨论主题发现**；人数与极性统计维持确定性回扫，主题本身不带立场。

**生成（每集一次 LLM 调用，离线增量）**：
- 输入：该集去重后的代表评论 ~30 条（评分/长度/时间/作者多样性采样，
  同作者 ≤2，总输入 ≤24k token，与 A3.3 同套采样器）；
- 输出：主题清单，每条 = 一句主题描述 + 方面标签（作画/演出/音乐/
  叙事结构/人物/改编/其他闭集）+ **2 条代表引文（原文逐字段落）**；
- 覆盖门槛：去重后评论数 <10 的集**不生成**，检索退回原文层，宁缺毋假；
- 作品级地图：集级地图聚合 + 跨集主题归并（同规则同校验）。

**引文逐字硬校验（出库门禁）**：
每条主题的引文必须是该集评论集合中的**逐字子串**（程序化 string
match），任何一条对不上即删除该主题。主题描述可以是幻觉，
**没有真引文支撑的主题不允许出库**——引文才是进上下文的东西。

**Schema**：

```sql
discussion_theme(
  id, subject_entity_id, episode_entity_id,  -- episode 为空 = 作品级
  theme_text, aspect_kind,
  quotes_json,            -- [{document_id, quote}]
  coverage_json,          -- 评论数/独立作者数/采样策略版本
  extraction_model, extraction_version, build_id,
  UNIQUE(subject_entity_id, episode_entity_id, theme_text, extraction_version)
);
```

**双向用途**：
1. **谈资**：话题锁定某集/某作品时，主题+引文注入证据包 scope-summary；
2. **路标**：主题文本进入检索（FTS + A3.2 向量范围），用户不说集数时
  可由"弯腰 solo 那段"命中 ep8 地图，反向锚定话题——与单集卡构成
  "官方事实"与"社区讨论"两条腿。

**注入纪律（写入 SOUL.md）**：讨论地图只作话题索引，任何具体说法必须
落在引文原文上；禁止把主题描述当作社区共识陈述。

**成本**：热作品（≤200 部）先行验证，约 2~3 千次调用；
全量 15,603 集 ≈ 1.5 万次调用（flash），验证通过后再铺。
**评测全程序化**：主题覆盖率 + 引文命中率，不评"写得好不好"。

---

## 3. A4 黑话发现管线

目标：识别"锯/团长/炮姐"这类不在任何数据源的民间黑话，分级补进别名表。
实测现状：宅方/小南极/石头门可识别，锯→空、团长/炮姐→错误命中。
**离线管线允许第三方库**【ADR-5】：`pyahocorasick` 直接引入；
word2vec 可用 gensim 也可用共现向量，实施者按效果自选。

### 3.1 六阶段（新模块 `src/mikirin/slang/`）

1. **候选生成**：字符 n-gram（n=1~5）+ 拉丁/假名/数字串；
   句法槽位正则（看__ / 追__ / __厨 / __第二季 等 30+）。绕开分词器。
2. **已知擦除**：entity_alias 编译 AC 自动机一遍扫描划掉已知名称；
   副产品 `cooccurrence` 表（A5 Pass A 复用同一次扫描）。
3. **可指称性评分**（纯统计）：min-PMI 凝固度 + 邻字熵 + 指称特征组
   （槽位命中率最强、上下文分布熵、用户广度、书名号缺失度）；
   阈值用 100 正/100 负人工样本校准（标注者：用户本人【ADR-6】）。
4. **分桶链接**：A 自指（锚点集中+与官方名互斥替代）；B 跨作品（向量近邻 top-3）；
   C 单字（滚/锯/咒）——**单独一桶，门槛最高，宁漏勿错**。
5. **LLM 仲裁**：选择题（候选 + 真实评论采样 + 假设实体简介），
   复用 organize-resolution 的缓存/容错/并发设施。
6. **分级入库**：高置信+用户广度>20 → `status='confirmed'`；中置信 →
   `status='provisional'`（权重 ×0.5）；`source_kind='mined'`。
   每月人工抽验 50 条（复用 review-resolution 界面）。

### 3.2 验收

- 黄金查询集（≥50 条，用户标注）Top-1 命中率 ≥90%，错误命中 = 0；
- 单字桶精确率 ≥95%；
- `search_anime("锯")` 命中《电锯人》，`search_anime("团长")` 候选含凉宫春日；
- 季度增量重跑分钟级（水位线记录）。

---

## 4. A5 图派生与隐式关系

### 4.1 确定性派生边（离线计算，物化回表）

`derived_edge(kind: staff_overlap / cv_overlap / tag_similarity / same_season)` +
`node_pagerank`。tag 相似度必须 IDF 加权（防"日常/恋爱"超级枢纽）。
在线只查表，不在线跑图算法。

### 4.2 隐式关系两段式

- **Pass A**：复用 A4 的 AC 扫描副产品 `cooccurrence`（≥2 实体共现文本）；
- **Pass B**：LLM 选择题判读（闭集关系词表 + 情感极性 + claim_quote），
  输出 `{src_entity_id, dst_entity_id, relation, sentiment, confidence, claim_quote}`；
- `implicit_edge` 边权 = COUNT(DISTINCT author_hash)，匿名不计数；
- 生成纪律：显式关系直接陈述，隐式关系必须带 attribution
  （"bgm 上有 N 个人觉得…"），claim_text 是唯一引用来源。

### 4.3 检索升级与强制对照【ADR-6】【ADR-10】

- 扩散在**按任务投影的子图**上跑（相似作品图/声优-角色图/staff 履历图），
  不在一张大图统一 PPR；PPR 重启 0.15、top-k 8 为初值；
- **对照实验**：30 个典型查询，简单检索 vs 图扩散并排匿名，
  **用户本人盲评定生死**（相关性+惊喜度两维度）；LLM 裁判仅用于调参期快速过滤；
- 向量检索已在 A3.2 限定范围落地【ADR-10 翻案】；A5 只在其基础上做
  图扩散与向量的融合实验，同样须用户盲评通过才上线。

### 4.4 验收

- derived_edge 同输入同 build_id 结果可复现；
- implicit_edge 抽样 100 条人工核查精确率 ≥85%，claim_text 可在原文定位；
- 对照实验报告存档 `data/manifests/quality/`。

---

## 5. B2 记忆管线 + 人格 + 软剧透纪律

### 5.1 人格【ADR-11】

`agent/SOUL.md` 由 Claude 起草、用户改定：**同好戏友型**——阅片量惊人、
有明确口味和立场、会犟嘴；立场必须标注为私货且服从事实纪律
（检索优先于立场）；用户说"别老提 X"必须记住（退订键）。
起草稿完成后经用户审定才进入 stable 层。

### 5.2 四层系统上下文（与 A3.4 一致，以此为准）

```
stable   — SOUL.md 人格 + 工具使用指南（长期复用，保前缀缓存）
context  — 环境信息、当前日期
topic    — community_report + 作品卡（A3.4；无锁定作品时为固定空围栏）
volatile — USER.md 画像 + 热记忆 + watch_progress 进度（每轮轻量重建）
```
其后接会话消息。

预取记忆包 `<memory-context>` 围栏注入，注入前 sanitize 防嵌套；
输出侧 scrub 剥离围栏（回归必须全绿）。

### 5.3 写入纪律（只两条路径）

1. 确定性状态更新（进度、明确偏好）：工具同步事务落 user.db（现状保持）；
2. session 结束统一结算：整 session 过 LLM 抽记忆卡片 → 冲突裁决
   （同实体旧卡片检索后判 新增/更新/失效/并存）→ 单 worker 串行落盘，
   幂等键 = source_session_id + 内容哈希。
   **模型分工：抽取用 flash、冲突裁决用 pro【ADR-12】**；
   每周人工抽查 20 条卡片（复用 review-resolution 界面），准确率 <90% 再升级。

### 5.4 软剧透纪律【ADR-9】

**不做检索层硬过滤**。watch_progress 注入 volatile 层，SOUL.md 加纪律：
"不主动讨论超过用户进度的情节；用户自己提起的除外；用户明确说
'这部无所谓'的作品视为解禁。"进度同时影响聊法（刚看第 3 集 vs 二刷）。

### 5.5 验收

- 说"我看到第 7 集"→ 下一轮 volatile 进度已更新；
- 纠正进度后旧卡片 valid_until 被填、不回溯出旧值；
- USER.md 超 1375 字符写入被拒并提示合并；
- 对刚看 3 集的用户，agent 不主动提第 8 集情节（人工抽查）。

---

## 6. B3 主动搭话【ADR-13】

- RSS 轮询（蜜柑镜像）追新：用户订阅的番更新 → 生成话题 →
  **下次上线时按概率投递**（不是每次都搭话，概率作为人格参数，初值 30%）；
- **总开关**：user 配置可整体关闭；
- **自动下载永久不做**（版权 + 误操作风险，与陪伴无关）；
- 消息队列存 user.db，投递后标记已读。

---

## 7. 更新节奏【ADR-14】

- BGM dump：**平时月更，新番季首月（1/4/7/10 月）周更**；
- `mikirin update --now` 手动立即更新；
- TUI 内 `/update` 设置频率（与 /models 同级，存 runtime 配置）；
- 每次更新 = 全量重建 + 增量消歧（规则层分钟级 + 新增候选进 LLM 队列）。

---

## 8. 模型与 provider 策略【ADR-15】【ADR-16】

- **provider 无关的 OpenAI 兼容接口**：provider/base_url/model/key 全部可在
  TUI `/models` 配置，存 `data/manifests/runtime_model_config.json`（不存密钥，
  密钥在 `.env`）；
- 默认 deepseek（chat=v4p / organizer=v4f），用户可指向任何兼容端点；
- **embedding 是第三个独立角色**（配置见 A3.2，不复用 chat provider），
  同样在 TUI 可配；
- 隐私底线：不追求"数据不出门"；密钥本地、user.db 独立可导出可删除。

---

## 9. 总顺序与回滚

```
A2 实体合并构建 ──────────────────────────── 发布 merge 版（不等爬虫）
A3 评论入库（爬虫完成后）────────────────── 发布 docs 版
A4 黑话管线（依赖 A3 语料）  ─┐
B2 记忆+人格（Agent 侧）      ├─ 可并行
A5 图派生（依赖 A3+A4）──────┘
B3 主动搭话（依赖 B2 人格与进度）
每次发布都是旁路构建 + current.json 原子切换；
验证失败：current.json 指回上一版本即回滚。
```

---

## 10. 架构决策记录（ADR）

| # | 决策 | 理由（防止翻案时忘记） |
|---|---|---|
| ADR-1 | 实体合并，BGM 规范侧，不做链接方案 | 一个角色一个实体的查询体验；错合并可拆分回滚 |
| ADR-2 | 冲突事实两边全留，展示 BGM 优先 | source_asset 天然共存；萌百 CV 中文译名是消歧救命的桥；证据比结论长寿 |
| ADR-3 | 硬门禁只用 78 人工；500 LLM 标注降为监控 | 标注集是规则层的"训练集"，拿它当硬门禁是循环论证 |
| ADR-4 | 拆分走决策文件重放，工具用到再做 | split JSONL 是契约；提前做工具会猜错真实需求 |
| ADR-5 | 第三方库：离线管线可引，运行时纯标准库 | 离线跑完就扔；运行时要每天启动，环境越薄越好 |
| ADR-6 | A4/A5 的效果裁判都是用户本人 | "搭子感"无客观指标；LLM 裁判评不出口味 |
| ADR-7 | 复核尾巴：different 全收、低置信 same 全拒、uncertain 挂账 | 合并要证据，否决可以便宜；错合并是唯一不可逆的伤 |
| ADR-8 | 萌百限定角色保留为独立实体 | 知识库差异化价值；来源标记可随时过滤 |
| ADR-9 | 剧透改软纪律，取消硬过滤与 content_scope | 单人工具做数据级剧透标注是过度工程；软纪律 5% 成本 80% 收益 |
| ADR-10（2026-07-29 翻案） | 限定范围语义向量检索直接进入 A3.2：单集卡+社区切片+简报；embedding 角色独立配置带指纹；向量本地两级索引（SimHash 粗排+量化精排），运行时零第三方依赖 | 原案假设语义查询是少数失败场景；实际上"哪一集/哪个番/模糊回忆"是闲聊核心高频需求且结构性无实体可锚。仍拒绝 embedding 成为所有知识的默认索引 |
| ADR-11 | A 型戏友人格（犟嘴、有立场、立场可退订） | 产品定义是陪伴闲聊；人格服从事实纪律 |
| ADR-12 | 记忆结算：抽取 flash + 裁决 pro | 裁决错了污染长期记忆，抽取错了代价低 |
| ADR-13 | 主动搭话带开关+概率触发；自动下载永不做；截图识别搁置 | 朋友是想起你才戳你，不是打卡机器人 |
| ADR-14 | 月更 + 季首周更 + 手动 + TUI 可调 | 数据变化形状是"季首剧变、季中平稳" |
| ADR-15 | provider 无关 OpenAI 兼容，TUI 可配 | 避免供应商锁定 |
| ADR-16 | 隐私=数据 ownership，不是数据不出门 | 密钥本地、user.db 可导出可删除 |
| ADR-17 | A3.1 确定性入库零 LLM 零 embedding（后被 ADR-10 翻案限定为仅指 A3.1）；语义理解以查询时生成与冷热胶囊为主，不做全量预计算 | 全量预计算烧钱且大半没人查；关系 95% 可确定性推导 |

---

## 11. 明确不做（本路线图 out of scope）

- 截图识别【ADR-13：无限期搁置】；
- embedding 扩展到实体层/全库向量库（ADR-10 翻案限定范围之外）；
- 观点簇/claim 全量预计算（ADR-17：查询时生成为主）；
- 作画层深化（atwiki 人名实体化、sakugabooru）——独立线，可随时插队；
- 自动下载（ADR-13）；
- Neo4j、agent 框架、向量数据库服务、Docker——永久不做（v1.2 已定）。
