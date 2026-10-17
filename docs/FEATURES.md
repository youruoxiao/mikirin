# 路线图功能怎么用

这里讲程序入口。人工标注、真实库上的效果和运行速度要单独验收，不能用测试全绿替代。
以下命令从项目目录运行；换资料库时在子命令前加 `--root <工作目录>`。
所有审核文件都可能包含原始评论或个人记忆，请留在本地，不要提交到公开仓库。

## 混合检索与 60 问评测

```sh
mikirin research-search "孤独摇滚第8集的演出有哪些争议" --mode evidence
mikirin research-search "那场 live 翻车的是哪集" --mode hybrid --vectors
mikirin tui --retrieval-mode evidence
mikirin retrieval-eval --output data/manifests/quality/retrieval-local.json
```

四档是 `fts`、`anchored`、`evidence`、`hybrid`。未传 `tui --retrieval-mode` 时，聊天仍用原有稳定路径。
`--vectors` 才允许命令行评测调用 embedding；`--disambiguate` 额外允许 organizer 在给定候选中选择章节，`--stance-classify` 允许它对给定原文标注支持/反对观点。结果按模型、接口和输入缓存，这些选项都可能产生费用。机器观点标签会核验连续引文，但不会标成人工核实。TUI 显式选择 `hybrid` 后使用已配置的双路向量索引。

混合检索依次处理明确作品/集数、对话焦点、别名和全文检索，再补语义候选与闭集消歧。
章节得分保留四个分量：单集卡 0.50、最佳评论切片 0.30、前三切片均值 0.10、词面匹配 0.10。
作品结果从章节聚合，返回来源、连续摘录和位置。缺索引、缺反方证据或缺作者信息会明确记录；不会编一个引用填满配额。

随包的 `resources/benchmarks/retrieval-60.json` 是版本化问集，不是假装已经标好的答案。
可以复制后填写 `gold`，同时在 `annotation` 中注明 `source: "human"` 和 `annotator`，再用 `--benchmark` 指向它。
缺少人工标准时，作用域准确率、证据召回和引用正确率保留 `null`；逐字校验仅证明引文来自原文，不证明它回答了问题。

匿名评测先打乱方案，再交给人评分：

```sh
mikirin retrieval-review prepare --report data/manifests/quality/retrieval-local.json --output-dir data/staging/retrieval-blind-new
mikirin retrieval-review score --blind data/staging/retrieval-blind-new/retrieval-blind.json --private data/staging/retrieval-blind-new/retrieval-private.json
```

评分者只看 `blind`，填写 reviewer、ratings 和 notes；不要改候选和引文。`private` 留给解盲，不随评分表发出去。汇总结果不会自动改变聊天默认方案。

## 会话记忆与可恢复结算

正常使用仍是 TUI `/settle`，或退出时确认结算。想将入队和执行分开：

```sh
mikirin memory-queue enqueue
mikirin memory-queue status
mikirin memory-queue run --limit 10
```

`enqueue` 默认取最新会话，也可以传 `--session-id`，只写本地队列，不调用模型。`run` 使用 organizer 抽取、chat 裁决，兼容这两个角色各自的接口配置。
长会话分块读取，不截掉尾部；抽取结果和逐卡处理记录会落盘。失败后重新执行 `run` 会接着处理，进程锁保证只有一个结算 worker。
候选内容按“会话 ID + 内容哈希”去重；旧卡失效、新卡写入和处理记录在同一个事务内。后续上下文只预取仍有效的卡片。

观看进度走同步工具写入，下一轮上下文立即刷新，不必等整段会话结算。`USER.md` 上限 1375 字符，超限拒绝写入。
剧透纪律是提示词约束，不是检索硬过滤；仍需要用真实对话做人工抽查。

每周审核：

```sh
mikirin memory-review prepare --week 2026-W42 --output data/user/reviews/2026-W42.json
mikirin memory-review score --input data/user/reviews/2026-W42.json
```

只填写卡片的 `label`（`correct` / `incorrect`）和 `notes`。没有 20 张完整人工标签就不算验收通过；完整样本低于 90% 才建议升级模型或规则。

## 黑话、共现和关系图

构建工具与分阶段恢复见 [scripts 使用说明](../scripts/README.md)。它们写旁路数据库，正式库不会因为跑评测而切换。
别名候选要经过闭集仲裁：已确认的高置信别名与临时别名分级入库，临时别名按半权重检索；未审核或已拒绝的挖掘结果不进入搜索。

```sh
mikirin graph-search "凉宫春日" --mode simple
mikirin graph-search "凉宫春日" --mode graph --database <旁路库> --build-id <构建ID>
mikirin p1-review graph-prepare --database <旁路库> --build-id <构建ID> --output-dir data/staging/graph-review-new
mikirin p1-review graph-score --blind data/staging/graph-review-new/graph-blind.json --private data/staging/graph-review-new/graph-private.json
```

图检索只查离线物化的任务子图，不在聊天时跑整图扩散。可选 `similar_work`、`voice_cast`、`staff_filmography`，默认 restart=0.15、top-k=8。
`graph-prepare` 默认用随包 30 问；只有加 `--vectors` 才请求 embedding。没有向量输入的结果不能当成完整向量融合实验。
盲评文件给人打相关性、惊喜度分；`private` 文件是解盲映射，不要提前交给评分者。评测不会自动开启融合路径。

其余审核入口：

```sh
mikirin p1-review slang-calibrate --input <100正100负标注.json> --version threshold-v1 --output <新报告.json>
mikirin p1-review slang-score --input <至少50问人工gold.json> --database <旁路库>
mikirin p1-review implicit-prepare --database <旁路库> --build-id <构建ID> --output <新审核.json>
mikirin p1-review implicit-score --input <填写后的审核.json>
mikirin p1-review aliases-prepare --database <旁路库> --month 2026-10 --output <新审核.json>
```

月度别名抽查取 50 条，隐式关系取 100 条；样本不足会如实报告，不补虚构数据。
隐式关系的作者数、逐字 claim_quote 和来源归因都保留；人工判断关系是否成立与程序检查引文是否逐字是两件事。

## RSS 主动搭话

默认关闭，只接受你明确提供的作品和订阅源：

```sh
mikirin rss subscribe "放映部" https://example.test/feed.xml
mikirin rss configure --enabled on --probability 0.30
mikirin rss poll
mikirin rss status
```

示例地址不是实际订阅源，需要换成自己的 RSS/Atom 地址。`subscribe` 不会自动联网；`poll` 才读取明确订阅的源，可以由你自己的定时任务重复运行。
同一更新不会重复入队。下次打开 TUI 时，只读取本地队列，按配置概率最多投递一条，不会为此暗中轮询。
话题不直接复述条目标题中的剧情，避免开场就剧透。TUI 里可以用相同的 `/rss ...` 命令，离线模式禁止 `poll`。

```sh
mikirin rss configure --enabled off
mikirin rss unsubscribe <subscription_id>
```

关闭后停止生成和投递，普通聊天不受影响。投递失败保留重试；成功显示后确认已读。
终端显示和 SQLite 提交不能组成同一个事务：如果恰好在显示后、确认前进程崩溃，重启可能再展示一次。不会把这个窗口说成“绝对只显示一次”。
这里只读取 feed 元数据，不访问附件、不下载视频、不调用 BT。
