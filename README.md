# 屿镜天文·暗星可见极限估算

以标准星场校准曝光为样本，按观测夜前向比较曝光对数和天空条件等特征，评估 V 波段极限星等，并据星等余量安排目标曝光。

## 模拟场景

[业务剧本](yangyang-business.md)记录模拟团队、上游观测、统计口径和交接去向。四个工作模块包括数据台账、分析任务、结果与复核、报表分发。数据从合成种子或 UTF-8 CSV 进入。分析员提交后，Python worker 对固化观测清单计算。负责人签核，随后可以下载结果或把报告定时投递到本地收件箱。

### 带修订号的更正

上游测光有时隔夜才核定标准星场的极限星等。台账按校准曝光编号保存每个修订，用 `revision`（正整数，留空即 1）区分版本，用 `received_at`（带时区的收到时间）记录入账时刻。晚到的更正按编号合并：同一编号只采用收到的最高修订，旧版不会盖过新版；完全相同的重试不会重复入账。原始记录视为在观测夜结束时（北京时间当日 24:00）收到；修订号大于 1 而未填收到时间时，以服务端接收时刻为准。

分析在发起时固化整份台账版本快照。前向验证的每折训练与最终留出训练只使用该段首夜之前已收到的版本，因此晚到更正不会污染更早的训练样本；各折验证、最终留出评分和排程拟合则使用发起分析时已收到的最新修订。使用同一快照重跑结果一致；新发起分析才会纳入其后收到的更正。结果页「分段版本核对」逐段列出各段实际用到的曝光版本、截止时刻和排除的帧数；若某段在首夜前凑不够六个训练夜，计算会失败并说明是哪几夜尚未收到有效版本。

## 启动

```sh
cp .env.example .env
docker compose up -d --build --wait
docker compose exec -T backend python -m scripts.seed_demo
```

浏览器打开 `http://127.0.0.1:18379`。API 健康检查位于 `http://127.0.0.1:19379/api/health`，数据库本地端口 `17879`。停机用 `docker compose down`。只有要重做演示数据时才加 `-v` 清除命名卷。

## 演示账号

| 角色 | 邮箱 | 演示密码 | 可执行动作 |
|---|---|---|---|
| 业务分析员 | `analyst@demo.local` | `Analyst-2026!` | 登记或导入样本、提交分析、下载内部结果 |
| 复核负责人 | `supervisor@demo.local` | `Supervisor-2026!` | 导入资料、签核或退回、安排或停止分发、查审计 |
| 只读查看 | `viewer@demo.local` | `Viewer-2026!` | 查看台账、已签核结果和本地收件箱 |

`.env` 以 `.env.example` 为模板。`FRONTEND_PORT`、`BACKEND_PORT`、`DB_PORT` 可做本机端口调整。Docker 私有子网可通过 `NETWORK_SUBNET` 覆盖。`JWT_SECRET` 和 `DB_PASSWORD` 为本地演示配置。`SEED_DEMO=false` 可跳过自动种子。所有公开端口绑定在回环地址。首次执行种子后，后续种子只填缺失观测，不覆盖已录入数据。

## 模块和本地开发

| 服务 | 技术 | 用途 |
|---|---|---|
| frontend | HTML、CSS、ES Modules、Nginx | 工作台、录入与报告 |
| backend | Python 3.12、FastAPI、SQLAlchemy | 校验、快照、业务 API 和 JWT |
| worker | Python 3.12、Redis | 计算排队、心跳和到时导出 |
| db | MySQL 8.0 | 观测、快照、签核、分发计划和审计 |
| redis | Redis 7 | 唤醒、心跳和登录限流 |

在 `backend` 中可用 `pip install -r requirements.txt` 安装本地依赖，设置 `DATABASE_URL`、`REDIS_URL` 和可写 `ARTIFACT_ROOT` 后运行 `uvicorn src.main:app --port 8000`，计算进程运行 `python -m src.worker`。前端修改在 `frontend/src`，用 `docker compose up -d --build frontend` 通过 Nginx 检查同源代理。业务输入定义在 `backend/src/domain/spec.json`，算法在同目录的 `analysis.py`。角色、任务状态和签核状态在后端 `backend/src/constants/enums.py` 和前端 `frontend/src/constants/enums.js` 共同定义。数据库表在 `backend/src/models/workflow.py`。

分发计划锁定特定已签核快照，按北京时间显示。收件箱可下载真实 CSV，当前收件组是本地交接标签。停机跨过多个周期时补送最早未投递的一期，然后追到下一个将来的周期。新快照需重新签核并建新计划。

许可证为 [MIT](LICENSE)。
