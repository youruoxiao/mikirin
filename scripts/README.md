# 可维护的本地工具

这里保存需要随 Git checkout 分发的脚本。原 `work/` 中的副本仅作为本地历史保留，
不再作为这些脚本或对应测试的代码依赖。`scripts/` 不是安装后的命令行入口；
请在仓库 checkout 中使用 Python 3.12+ 运行，无额外第三方依赖。

## 运行方式

以下示例从仓库根目录执行。四个工具都支持 `--help`，也支持
`python -m scripts.<模块名>`。直接指定脚本绝对路径时可以从其他目录执行；
P1 工具会从脚本位置找到本仓库的 `src/`，无需先进行 editable install。
除 `build_p1.py` 默认项目根目录外，命令中的相对输入/输出路径相对于当前工作目录。

### P1 分阶段构建

```powershell
python scripts/build_p1.py --help
python scripts/build_p1.py prepare --root . --version local-p1 --build-id local-p1
python scripts/build_p1.py graph --root . --build-id local-p1
```

需要自行准备 `db/current.json` 指向的现有内容数据库；空 checkout 只提供代码，
不包含生产数据库。进度保存在 `data/manifests/p1/<build-id>/progress.json`，
旁路数据库写入 `db/`。相同 build ID 用于恢复执行。支持 `prepare`、`slang`、
`graph`、`pagerank`、`implicit`、`validate`、`publish`、`all` 阶段。

`slang` / `implicit` 的模型仲裁可能访问网络并产生费用。凭据来自环境变量
（默认 `SILICONFLOW_API_KEY`）或项目根目录的本地 `.env`；不要将密钥写进脚本。
可选的 `data/manifests/runtime_model_config.json` 缺失时使用默认配置。
只有 `publish` / `all` 显式传入 `--publish` 且通过校验及人工验收门禁，才会切换
`db/current.json`。上述示例不发布数据库。

### 季度增量与冻结跨作品候选

首次创建 build 时指定增量基线与可选 neighbors 文件，后续同 build ID 自动恢复配置：

```powershell
python scripts/build_p1.py prepare --root . --version local-p1-next --build-id local-p1-next --run-id slang-next --incremental-base-run-id slang-baseline --vector-neighbors-file data/staging/slang-neighbors.json
python scripts/build_p1.py slang --root . --build-id local-p1-next
python scripts/build_p1.py graph --root . --build-id local-p1-next
```

基线 `slang-baseline` 必须已完成且存在于 `current.json` 指向的源数据库中，随旁路库复制；
处理器版本和其余发现配置必须兼容。若基线用过非默认限制参数，应在首次 `prepare` 同样传入。
没有兼容基线时省略 `--incremental-base-run-id`，建立首次完整基线；不要将新 run 作为自身基线。
这不是定时任务，季度执行由用户或外部调度触发；`slang` 仍可能进行付费模型仲裁。

neighbors 文件是 UTF-8 JSON 对象，键为黑话，值为冻结的实体候选数组，例如：

```json
{"锯": [{"entity_id": "subject:replace-with-real-id", "score": 0.97, "evidence": {"index_fingerprint": "your-index-fingerprint"}}]}
```

示例 ID 必须换成实际数据库实体 ID；`score` 必须是 [0,1] 有限数，`evidence` 可省略。
不调用向量 API：发现阶段使用文件里的候选并取 top-3。规范化后重复的黑话键或重复实体会被拒绝。
两个参数亦支持 `--slang-incremental-base-run-id` / `--slang-vector-neighbors-file` 别名。

`progress.json` 保存完整候选内容、源文件绝对路径、文件字节 SHA-256、规范化内容 SHA-256。
每次续跑（即使省略文件参数或阶段已经完成）都校验文件和保存内容；文件缺失、修改（包括空白变化）、
修改基线或中途新增文件输入都会拒绝，不会覆盖旧进度。保持源文件不变；若仅移动文件，续跑时
重新传 `--vector-neighbors-file <新路径>`，新文件必须字节一致。更换内容请使用新的 build ID/run ID。

`graph` / `pagerank` / `all` 会识别旧的作品投影阶段记录并重建异构任务图。
声优任务支持 person/character/subject，staff 履历支持 person/subject；`similar_work` 不变。
构建仍为旁路，不传 `--publish` 不切换当前库。

### 生成人工复核包

```powershell
python scripts/prepare_p1_manual_review.py db/content.vlocal-p1.db local-p1 slang-local-p1 data/staging/p1-review
```

参数依次为旁路数据库、build ID、slang run ID、输出目录，可选 `--seed`。
只读数据库，先检查数据充分性，再写入固定抽样的 CSV/JSON；人工标签保持空白。
同名复核文件会被替换，请使用新的输出目录以保留已填写的标签。

### 提取讨论地图及质量校验

```powershell
python scripts/extract_discussion_maps.py --base data/staging/discussion_maps/my-input --outputs data/staging/discussion_maps/my-output
python scripts/check_theme_quality.py data/staging/discussion_maps/maps.jsonl
```

提取器不调用模型。`--base` 下需有 `inputs/*.json`，每个文件包含
`episode_entity_id` 及 `comments` 数组（元素含 `comment_id`、`content`）。
使用 `--batch` 时还需 `batches/batch-*.jsonl`，每行的 `file` 指定 inputs 中的文件。
输出为逐文件 JSON；默认跳过已有输出，可用 `--overwrite` 或
`--overwrite-extractive` 控制覆盖。建议显式指定路径：省略参数会使用历史的
`theme-v2-full-20260801` / `theme-v2-extractive-20260801/outputs` staging 目录，
这些数据不随仓库分发。

`check_theme_quality.py` 同时提供提取器使用的质量规则，无需任何 `work/` 文件。
其命令行输入为 JSONL（一行一个含 `themes` 的地图对象），不是单个多行 JSON；
如需检查提取结果，应先将对象整理成 JSONL。报告 `quality_report.json` 和
`rejected_themes.jsonl` 写到输入文件的同级目录，会替换同名报告。

## 离线测试

在仓库根目录运行（PowerShell）：

```powershell
$env:PYTHONPATH = "$PWD/src"
python -B -m unittest tests.test_build_p1_script tests.test_prepare_p1_manual_review tests.test_extract_discussion_maps -v
```

测试使用临时 SQLite/内存样例并模拟模型边界，不需要本地数据库、`.env`、
API 密钥、网络或旧 `work/`。启动回归检查还会复制 `src/` 和 `scripts/` 到临时目录，
禁用环境导入路径和已安装包，验证直接执行及模块执行的全部 `--help` 入口。

只维护本目录的代码和说明；数据库、模型缓存、评论数据、生成报告及密钥应留在
现有忽略规则覆盖的 `db/`、`data/staging/`、`data/manifests/`、`data/llm_cache/`
或本地 `.env` 中，不要加入 `scripts/` 或提交到 Git。
