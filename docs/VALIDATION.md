# 本机验收记录

日期：2026-11-02。系统：Windows；GPU：NVIDIA GeForce RTX 5060 Ti（约 16 GB）。真实训练解释器为现有 `env_ame`，Isaac Sim 独立安装目录为 `C:/isaacsim`。工作台依赖安装在自身 `.venv`。

## 已完成

- 前端 TypeScript 检查与 Vite production 构建成功。
- 20 项自动测试通过，覆盖参数与路径、网格、串行接续、失败暂停、源码漂移、依赖排序、停止、进程身份、配置 API、跨项目环境构造、工作流导入依赖重映射、显式外部进程等待、停止服务后训练保持运行与恢复不重复启动。
- 独立服务运行于 `127.0.0.1:8765`，前端构建由同一服务提供；状态 SSE、日志分页、训练产物接口可用。
- 使用真实 Go2 任务运行三项串行实验，每项 4 个环境、2 次更新、保存间隔 1，全部成功；由同一工作流自动接续。

| 实验 | 方式 | 加载 checkpoint 编号 | 本次更新数 | 最终编号 |
|---|---|---:|---:|---:|
| qa_new | 从头训练 | — | 2 | 1 |
| qa_resume | 恢复模型、优化器与编号 | 1 | 2 | 2 |
| qa_warm_start | 仅加载权重并重置 | 2 | 2 | 1 |

三项实际生成的 `params/env.yaml`、`params/agent.yaml` 均已读取核对：

```yaml
scene.num_envs: 4
rewards.feet_gait.weight: 0.15
reward_gates.groups.gait_guidance.end_iteration: null
reward_gates.overrides.lin_vel_z_l2.terrain_scales.discrete: 0.5
agent.save_interval: 1
```

最终 checkpoint 由训练 Python 使用 `torch.load(weights_only=True)` 在 CPU 上检查，包含模型与优化器状态；三项产物均位于独立工作台 `outputs/training/<attempt-id>/...`，而非训练项目源码目录。

原始验收运行记录：`data/real-acceptance.json`。对应工作流名称“本机验收 · Go2 真实短训练”，可以在页面查看。每项的 `recipe.json`、`console.log`、`progress.json` 与 `result.json` 保存完整依据。实际 YAML 检查结果及后续 HTTP 验收见 `data/final-verification.json`。

## 发现并解决的问题

- 本机 Conda Python 没有自动继承 Isaac Sim 激活环境，导致 `SimulationApp` 导入失败。项目档案增加独立安装目录与显式环境变量，构造训练子进程环境后，真实配置导出和训练通过。
- Hydra 对 float 默认值的类型检查会拒绝整数零，命令构建与校验将奖励权重规范化为 `0.0`。
- 停止工作台服务需要只终止 API 与 worker 自身，避免递归终止独立 supervisor。已修正并以持久化进程测试验证。

## 验证限制

- 浏览器自动化被系统终止：无法足够可靠地识别当前浏览器 URL。已停止一切界面控制，因此没有完成两个主题下的逐项点击、响应式截图与视觉比对。编译成功和 HTTP 检查不能替代此项交互验收。
- 真实验证是短训练接入验收，不证明长训练收敛或奖励设计效果。
- Linux / 其他项目尚未现场执行。提供跨平台启动代码、安装锁文件、档案与工作流迁移，以及明确的适配器接口；仍需在目标机器做短训练验证。
- 自动测试出现一条 Starlette TestClient 对 httpx 的弃用提示，未影响测试或运行服务。
