# 给 Mikirin 准备数据

日常试用用 `mikirin --root ./demo-space demo` 就够了。这页是为要导入真实资料的人留的。

## 工作目录

所有命令都支持在子命令前传 `--root <目录>`。先初始化：

```sh
mikirin --root ./my-library init
mikirin --root ./my-library status
```

`data/raw/` 保存原始证据，`staging/` 保存规范化中间结果，`rejected/` 保存失败记录，`manifests/` 保存队列和构建状态。`db/current.json` 指向当前发布库。构建期间保留独立文件，校验后才切换指针。

## Bangumi Archive

自行取得允许使用的 Archive ZIP 后，先检查内容，再归档：

```sh
mikirin inspect-bgm-archive <archive.zip> --count
mikirin ingest-file bgm_archive <数据版本> <archive.zip> --license-id bgm-archive
mikirin stage-bgm-archive <archive.zip> <数据版本> <source_asset_id>
mikirin build-bgm-content <数据版本> <asset_manifest路径> --version <数据库版本>
```

`source_asset_id` 和 manifest 路径来自归档结果，不是随意填写的文件名。大构建可加 `--defer-publish`，后续用 `finalize-content` 补源、建索引并发布。请先看对应命令的 `--help`；真实数据可能需要数十 GB 磁盘和较长处理时间。

## 可选来源

```sh
mikirin fetch-moegirl
mikirin stage-moegirl <萌娘提交版本>
mikirin crawl-atwiki --refresh-queue --discover-only
mikirin crawl-atwiki
mikirin stage-atwiki <快照版本>
```

Bangumi 社区内容单独采集和构建：

```sh
mikirin discover-bgm-sample --start-year 2011 --end-year 2025 --yearly-limit 50 --rank-limit 250
mikirin seed-bgm-content
mikirin crawl-bgm-content
mikirin bgm-content-status
mikirin stage-bgm-web <快照版本> --require-complete
```

采集会访问第三方站点。默认限速和失败退避用于避免持续重试；不要为了赶进度去绕过站点限制。作者身份在本地加盐哈希，社区内容默认标记为不可再分发。能在本地分析，不表示能随仓库发布。

## 模型整理和实验模块

角色消歧先用规则层缩小范围，再用闭集模型仲裁：

```sh
mikirin resolve-rules --run-id rules-local
mikirin organize-resolution --candidate-file <needs-llm.jsonl> --run-id review-local
mikirin review-resolution --help
```

模型请求包含候选实体和评论/资料摘录，会发送给配置的服务商并消耗额度。结果保存在本地缓存和审核区。正式构建前仍要检查证据和质量门禁。

黑话发现和关系图的构建、人工审核工具在 [scripts](../scripts/README.md)。它们属于实验管线，不是新安装时自动启动的任务。人工阈值校准、黄金查询和盲评没有完成时，不能以代码测试通过替代效果验收。

历史运行记录见 `SPEC-ROADMAP.md` 和 `TODO-REMAINING.md`。记录里的本地路径、旧模型名和数据库数量描述的是当时的实验，不是此源码包自带的资源。
