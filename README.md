# Mikirin（ミキリン）

萌娘百科结合 Bangumi 的动画知识库 Agent。先查本地资料，再让模型接着聊；作品名、别名、章节、看过的进度和聊天记录都放在自己电脑上。

起因很简单：想聊某一集的时候，通用模型常常只知道整部番的大概，具体桥段却接不上。于是先做了一个 SQLite 知识库，后来又陆续加了评论检索、会话记忆和数据整理工具。现在它还是个人项目，最适合喜欢折腾本地资料库的人。

## 先玩一下

需要 **Python 3.12+**，自带 SQLite 需支持 FTS5/trigram。官方 Python 的常见发行版可以直接用。运行部分没有第三方 Python 依赖。

在源码目录里：

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 用：.\.venv\Scripts\Activate.ps1

python -m pip install -e .
mikirin --root ./demo-space demo
mikirin --root ./demo-space search "放映部"
mikirin --root ./demo-space tui --browse-only
```

你会搜到「屋顶放映部」。演示库只有三部**虚构作品**和九个章节，笔记也是专门写的样例，不是网上抓来的评论。它不用 API key，不下载数据；重复执行 `demo` 会复用样例库，遇到已有正式库则提示换目录。

终端里可以输入：

```text
/search 末班猫
/status
/help
/quit
```

`--root` 是保存数据库和个人记录的工作目录，放在子命令前。可以同时保留一个演示目录和一个正式资料目录。

## 接上聊天模型

先把 `.env.example` 复制为工作目录下的 `.env`（上面的例子就是 `demo-space/.env`），填上所用服务的 key，然后去掉 `--browse-only`：

```sh
mikirin --root ./demo-space tui
```

用 `/models` 查看配置。聊天和资料整理是两个独立角色：`chat` 和 `organizer`。内置默认配置是 DeepSeek；模型名保留为配置项，实际可用型号请以服务商账户为准。也可以直接配置兼容接口：

```text
/set chat https://your-provider.example/v1 your-model chat-completions MIKIRIN_API_KEY
```

这里的最后一项是环境变量名，**不要在命令里填密钥本身**。配置保存在工作目录的 `data/manifests/runtime_model_config.json`，可以随时换。

普通输入会发给聊天模型，并带上检索命中、会话上下文以及相关个人记忆。`/organize` 和会话结算也可能调用付费 API；`--browse-only` 用于纯本地浏览。项目不会随源码附送 API 额度或真实数据库。

## 现在能做什么

- 按作品名、角色名和别名搜本地知识库，查询章节和相关文档。
- 终端聊天、流式回复、会话保存与分叉；用结构化记录保存观看进度。
- 将取得的源数据归档、解析、校验，再构建独立的数据库版本。
- 整理角色消歧候选，保留原文、模型结果和人工审核记录。
- 会话结束后按队列结算记忆，支持冲突历史、失败恢复和人工抽查。
- 明确订阅 RSS 后，在上线时按概率提醒更新；默认关闭，不下载动画。

评论主题、向量检索、黑话发现和关系图也有实现，不过它们需要额外构建的数据。**有模块不等于默认已经启用**：演示版不包含全量评论、embedding 或图索引，P1 的人工标注和盲测也尚未完成。完整路线和历史验收记录放在 [docs](docs/TODO-REMAINING.md)，不拿它们当现成功能宣传。

混合检索四档、图谱实验、记忆队列、RSS 订阅，以及 60 问/30 问匿名评测的用法见 [功能使用说明](docs/FEATURES.md)。实验检索需显式选择，不会因代码写好就自动替换现有路径；评测工具也不会替用户填写评分。

## 换成自己的资料

真实数据的准备比较重，建议先把演示跑通。Bangumi Archive、萌娘条目和作画 Wiki 的导入说明在 [数据管线说明](docs/DATA_PIPELINE.md)。采集脚本需要网络，模型整理需要对应 API；离线搜索本身不需要。

数据库不放进 Git 历史。大文件发布方式、当前数据清单和公开前的检查见 [数据库分发方案](docs/DATABASE-DISTRIBUTION.md)。

```text
src/mikirin/       程序、检索、聊天与数据管线
src/mikirin/agent/ 模型调用和工具循环
scripts/          构建、审核、源码导出等维护工具
tests/            不联网的回归测试
agent/            可编辑的聊天风格和工具说明
docs/             构建说明、历史规格和待办
```

每个工作目录里的 `db/` 存数据库，`data/` 存原始资料、缓存、会话和个人记忆。请自行备份；这些文件不是源码的一部分。`agent/SOUL.md` 和 `agent/TOOLS.md` 可以覆盖随安装包提供的默认提示词。

## 开发和打包

```sh
python -m pip install -e .
python -m unittest discover -s tests
python -m pip wheel . --no-deps -w dist
python scripts/export_source.py
```

最后一条生成 `dist/mikirin-source.zip`，只收录代码、测试和文档，排除 `.env`、本地数据库、评论数据、向量文件及 Git 历史。可以解压后创建一个新仓库上传到 GitHub。开发仓库的旧历史曾包含生成数据，不能靠补一条 `.gitignore` 就把它们从历史里抹掉。

附带的 CI 配置在 Windows / Linux 上跑测试，并检查 wheel 安装；向 GitHub 写入该配置需要登录凭据具备 `workflow` 权限。数据源有各自的使用和再分发限制，参见 [第三方说明](THIRD_PARTY_NOTICES.md)。源码目前还没选定开源许可证；公开仓库不等于给第三方数据或代码额外授权。
