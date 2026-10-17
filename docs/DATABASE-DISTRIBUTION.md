# 数据库如何分发

调查日期：2026-10-18。源码仓库：<https://github.com/youruoxiao/mikirin>。
本文是分发方案，不代表数据库已经上传，亦不代表数据已取得再分发许可。

## 本地资产清单

大小使用十进制 GB；指针和状态来自本地只读检查。

| 内容 | 大小 | 用途与处理方式 |
| --- | ---: | --- |
| `db/content.v2026.08.01-a36.db` | 11.58 GB | 当前已发布内容库，优先作为运行数据候选 |
| `db/content.v2026.08.02-p1.db` | 30.10 GB | P1 旁路构建，自己的 build 仍为 `building`，不能冒充已验收正式版 |
| 当前向量 bundle `d1b045…d577` | 6.09 GB | 与 A3.6 内容库绑定；包含卡片、社区切片、insights 三类索引及 manifest |
| 旧向量 bundle `a536f7…ea10`、`f518df…9a31` | 合计 6.50 GB | 历史备份，首次运行不需要全部下载 |
| `data/llm_cache/embeddings/cache-v2.sqlite` | 5.66 GB | 构建缓存，运行不是必需；不可无审查整包公开 |
| pipeline / crawl queue SQLite | 约 25 MB | 本地构建和采集状态，不属于运行时知识库 |
| `db/user.db` | 约 2.27 MB | 个人记忆和会话数据，排除公开分发 |

两份内容库合计 41.68 GB。当前正式内容库与匹配的向量 bundle 合计约 17.66 GB，
这是未压缩的候选运行包；压缩后体积须实测，不能据此保证下载大小。

## Git、LFS 与 Releases

- 不把 SQLite 或向量二进制提交到普通 Git；源码与大数据应独立更新。
- Git LFS 按账号配额计存储与下载流量；GitHub Free/Pro 当前包含各 10 GiB，
  超额是否计费或被阻断取决于预算设置。整库多版本会重复占用存储，不作为本项目首选。
- 采用 GitHub Releases 附件：单附件必须小于 2 GiB，单 release 最多 1000 个附件。
  官方文档目前未设置 release 总体积或下载带宽上限；这不免除平台条款及数据权利要求。
- 实际上传采用 Python 标准库 gzip+tar、256 MiB 小分卷，方便断线重试且远低于单附件边界。一个数据版本对应一个 release，
  附带每卷 SHA-256、整体原文件 SHA-256、构建状态及恢复说明。

依据：[Releases 限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)、
[LFS 配额与计费](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。

## 公开前的数据范围检查

两份内容库各有 1,102,297 条 `community_document`，现有记录全部为 `redistributable=0`。
代码核对发现 `bgm_web_staging.py` 对该字段统一写入 `False`，加载器缺省也为 `False`。
它是保守的导入标记，不是逐条权利调查结论，不能单凭它断言禁止再分发。
同样，单纯将字段改为 1 也不构成许可证明。公开可访问不应直接标成 CC0 或公有领域。
萌娘、Bangumi、作画 Wiki 及社区评论须分别核实来源条款、署名和再分发范围。

已查到的来源说明：[萌娘用户协议第 10.2 条](https://app.moegirl.org.cn/usercontract/index.html)
说明除特殊注明外的用户文本采用 CC BY-NC-SA 3.0；这是附条件许可，不是公有领域声明。
[Bangumi Archive](https://github.com/bangumi/Archive) 说明其导出 wiki 数据；
这份说明不能直接当成另行采集的社区长评/评论的统一许可。
这些核对不构成对所有记录的最终法律判断，也不改变本地检索用途。

社区文本是 Agent 对话检索、观点分析和原文引用的正式数据来源；本地原文与检索功能保留，
源码发布不执行删库、不移除召回，也不把原文替换为仅有模型摘要。
公开分发需按来源说明许可、署名及适用条件；若选择另做脱敏/删减发布副本，
还需处理引用摘录、FTS、discussion maps、briefs 等派生内容，重建匹配的向量 bundle。
不能只删一张原文表，然后把仍含摘录或旧指纹的索引上传。
经过范围变更后的数据库必须有新构建 ID，并重新验收，不能冒用当前发布指针。

若需求只是自己跨设备备份，可另用私有存储；不要把公开仓库的 release 当作私有附件空间。
不上传 `.env`、`user.db`、个人会话、密钥、模型调用缓存和采集队列。

## 实施顺序

1. 先确定公开数据范围或私有备份目标。本次选择正式 A3.6 库、匹配向量和独立的 P1 备份，保留社区原文；排除个人数据和构建缓存。
2. 停止目标库写入，或使用 SQLite backup API 创建一致快照；不能直接忽略 WAL 复制正在写入的主文件。
   在快照上执行完整性检查，并记录所有 build 状态、schema、来源和许可信息。
3. 运行包同时携带准确的内容库文件名、`db/current.json`、向量 bundle 和对应的向量 `current.json`。
   不把未通过验收的 P1 库替换为默认库。纯备份若保留 P1，明确标记为未完成构建。
4. 使用仓库提供的快照、压缩、校验工具（当前明确选择 A3.6、P1 与匹配向量）：

   ```powershell
   python scripts/database_release.py prepare --root . --output dist/data-release-new --part-mib 256
   python scripts/database_release.py verify dist/data-release-new/manifest.json
   ```

   无需安装 7-Zip；三个包独立压缩。恢复工具只写全新目录，支持仅选正式库和向量，不必下载 P1 备份。
5. 生成逐卷及原文件 SHA-256。对一致快照执行完整 SQLite 检查，对压缩包完整流式解压并逐文件校验哈希，
   核对发布指针与向量指纹；发布端不必为验证再落地一份几十 GB 的副本。恢复端只向全新目录写入。
6. 创建草稿数据 release，逐卷上传；中断后检查服务器资产大小和 digest，跳过已核对相同的附件，
   不盲目覆盖已有附件。全部上传并校验完成后才发布。
7. 下载者收齐所选包的全部分卷和 `manifest.json`、`database_release.py`，执行 `restore` 恢复到新工作目录，核验后启动。
   不覆盖已有 `user.db`，也不自动覆盖用户现有的正式知识库。

源码仓库与数据库 release 可独立迭代；不为一次数据更新重新上传所有历史数据库。
