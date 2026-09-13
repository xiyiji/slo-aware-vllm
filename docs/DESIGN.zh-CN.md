# 第二阶段：面向延迟目标的 vLLM 调度参数优化

## 1. 为什么值得单独做成项目

第一阶段解决的是“一个请求能否从网页到达真实 GPU，并拿到正确响应”。第二阶段解决的是“很多请求同时到达时，在延迟要求内能够完成多少请求”。后者需要受控实验，不能仅凭使用了 vLLM、异步代码或者 GPU 就宣称性能优化。

本项目优化对象是 **vLLM engine scheduler 的配置**，不是修改 CUDA kernel，也不是发明新调度算法。单卡、单模型、固定工作负载，让老师和面试官能复现并检查每一个数字。

## 2. 目标、非目标和验收

目标：在同一 RTX 4090 上，为 Qwen2.5-7B-Instruct 找到比串行 sequence 基线更合适的配置，以 SLO goodput 为主要指标。

非目标：多 GPU 扩展、模型质量优化、生产容量承诺、跨硬件排名、自动扩缩容、GPU kernel 优化、成本最优云资源调度。

验收不是“页面能打开”，而是：

1. token 计量的 CPU 单元测试通过；缺失 usage 不允许猜测。
2. GPU 返回的输入/输出 token 数满足固定长度契约。
3. 每组保留原始请求记录、实际配置、环境信息、错误和设备采样。
4. 初筛和正式实验分开。冻结候选后，对基线和候选分别做至少三次正式重复。
5. 汇总程序能够从原始数据重新生成表格和图。
6. 恢复原演示服务并重新验证公网调用。

## 3. 两层 batching 的边界

| 机制 | 工作位置 | 聚合对象 | 本实验是否研究 |
|---|---|---|---|
| Gateway micro-batching | HTTP 网关 | 等待处理的请求 | 否，实验绕过网关 |
| vLLM continuous batching | GPU 推理引擎调度器 | 每次迭代的活跃 sequence/token | 是 |
| Prefix caching | 引擎 KV cache | 相同前缀的 KV tensor | 本轮关闭，避免混淆 |
| Gateway response cache | 网关内存 | 已完成的文本响应 | 不经过该路径 |

`max_num_seqs=1` 只能称为“串行 sequence 基线”，不能说关闭了所有 GPU batching。即使只有一个 sequence，prefill、矩阵运算和 CUDA graph 等优化仍可能存在。

## 4. 组件设计

- `experiment.py`：检查端口、保存环境、启动单个 loopback vLLM 服务、等待健康、采集遥测、运行测试、清理自身进程。
- `measure.py`：固定 token ID 数据集、固定种子 Poisson 到达、异步 SSE 解析、逐请求计量与 JSONL 落盘。
- `sweep.py`：串行启动候选配置，避免两个 7B 引擎挤在同一张卡。
- `analyze.py`：校验请求数量，读取精确 token 数和遥测，生成 JSON/CSV/PNG/Markdown。
- GitHub Actions：无需 GPU 的计量契约测试。绿色 CI 不等于 GPU benchmark 已通过，二者证据分开。

正式测试使用 vLLM 自带 OpenAI-compatible server，移除公网、Render response cache、Ray autoscaling 的影响。原 Ray Serve 集成仍用于真实应用演示。

## 5. 实验变量与控制

自变量：`max_num_seqs` 取 1/8/16/32/64；`max_num_batched_tokens` 取 2048/4096/8192。

控制变量：同一张 GPU、同一模型权重、bfloat16、0.9 显存比例、4096 上下文上限、prefix cache 关闭、chunked prefill 开启、temperature=0、输入256 token、输出128 token。

输入使用固定种子的 token ID 数组，输出通过 `ignore_eos=true` 固定长度。它适用于调度实验，不用于回答质量或真实业务语义评价。每次记录 workload SHA256；同一 repetition 的两个配置使用同一 seed。

所有候选采用同一 **2 requests/s** 的到达过程。初筛每组64个请求；正式比较每组128个请求、至少三个 seed。每次3个单独 warm-up 请求，不计入结果。首次启动的模型下载、加载、编译不计入吞吐量。

为了控制租赁开销，先在4096 token budget下筛选五个 sequence 候选，再对最佳 sequence 试另外两个 budget。没有运行的组合明确标记为未测试，不声称完成全部15格搜索。

## 6. 时间、token 与 SLO 定义

- TTFT：客户端开始发请求到第一个非空内容到达的时间。role-only、usage-only 帧不算首 token。
- E2E：发请求到完整流结束，包括最终 usage 帧。
- TPOT proxy：`(最后内容到达 - 第一段内容到达)/(completion_tokens - 1)`。这是平均每个输出 token 的客户端时间，不是逐 token 的真实间隔分布。
- output tokens/s：所有有效成功请求的精确输出 token 总数 / 整批含 drain 的墙钟时间。
- requests/s：有效成功请求数 / 墙钟时间。
- goodput：同时满足 `TTFT <= 1秒` 和 `E2E <= 10秒` 的成功请求数 / 墙钟时间。

SSE chunk 可能含多个 token，甚至没有内容。因此计数必须来自引擎 `usage.completion_tokens`。HTTP 200 但缺少 `[DONE]`、usage、finish reason，或者 token 长度不符，都不是成功样本。

请求总 deadline 为60秒。失败与超时保留在 attempted 数和整体时间分母里，不能删除失败后只展示成功部分。成功请求的分位数同时报告样本数；失败率高时必须一起阅读，避免幸存者偏差。

## 7. 遥测与可解释性

每秒采集 `nvidia-smi` 的 GPU utilization、显存、功耗，并采集引擎 metrics。只统计测量窗口内的采样，不把预热或闲置阶段混进平均 utilization。

记录 KV-cache usage、running/waiting requests、preemption counter；如果该运行时没有暴露指标，记为 unavailable，不写0。OOM保留服务端日志证据；没有搜到 OOM 文本不代表所有错误原因已排除。

预期解释链条：sequence=1 造成引擎排队 → TTFT 增大 → SLO miss；适当增加活跃 sequence → GPU 可以处理多条 decode → 同等负载下排队缩短。是否成立以数据为准。低到达率下 token budget 差异可能很小，应如实报告“在该负载下无明显优势”。

## 8. 选择与统计

初筛按 goodput 排序，错误率与 p95 TTFT 作为约束/次级指标。相近候选优先较小、较简单的配置，而不是把微小噪声宣传成最佳配置。

正式测试不能反过来重新选参数。三次重复给出逐次结果与均值/范围；三次不足以支持强统计显著性或生产 p99 保证。吞吐提升公式为 `tuned/baseline - 1`，但若 baseline goodput 为0，禁止报告无限倍提升，应报告绝对变化。

初筛中发现的计量或环境问题必须记录。如果修复改变测量定义，正式两组必须使用同一修复后的代码。旧初筛不可冒充正式对照。

## 9. 部署、回滚和成本

实验前确认原服务 PID、代码版本与监听地址，只关闭自己的已确认推理 deployment，不停止其他用户任务。实验服务只监听 Pod 内 loopback 18001，不将管理端口暴露公网。

实验结束拉取已测试的 Ray 集成版本，启动绑定 `0.0.0.0:8000` 的 Serve proxy，然后部署模型。验证 `/healthz`、真实非流式推理、真实 SSE、Render 转发、浏览器 Chat。

不自动购买新 GPU。Pod 是否停止由用户选择；视频和已保存数据不依赖 GPU 永久在线。视频展示真实操作画面，但不是性能计时来源。

## 10. 公开表述边界

可以写：在固定模型、固定 token 长度、固定到达率、单 RTX4090 的受控实验中，比较 scheduler 参数，报告实测 goodput/TTFT/吞吐量。

不能写：生产级高并发保证、普遍适用于所有负载、CUDA kernel 加速、降低某个百分比成本，除非另有相应实验。后续扩展可以加入长短请求混合、不同到达率、prefix-cache 命中率控制、更多重复，以及基于请求长度的 admission control。
