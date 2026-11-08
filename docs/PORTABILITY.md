# 迁移与项目适配

## 迁移到 Linux 服务器

1. 复制工作台的 `backend`、`frontend`（含 package-lock.json，可不带 node_modules）、`adapters`、`tests`、`docs`、`design`、根目录 Python 打包文件、锁文件和启动脚本。不要复制 Windows `.venv`、node_modules 或正在使用的 SQLite 数据库。
2. 在服务器准备可正常启动目标任务的 Isaac Lab / RSL-RL 环境、GPU 驱动和 Node.js，然后执行 `bash setup.sh`。服务环境重新创建，原有训练环境不变。
3. 执行 `bash start.sh`，在本机建立转发：

```bash
ssh -L 8765:127.0.0.1:8765 -p <SSH端口> <用户名>@<服务器>
```

4. 浏览器访问 `http://127.0.0.1:8765`，导入项目档案后修改根目录和 Python 为服务器绝对路径，按安装方式填写 Isaac Sim 目录，保存并刷新配置。
5. 导入工作流。依赖前项最终 checkpoint 的关联自动重建；固定文件路径要先在 JSON 中改成服务器实际路径，否则导入会按校验规则拒绝不存在的文件。
6. 先跑 4 个环境、2 次更新的小实验，再开始完整计划。

工作台只监听回环地址，不提供公网账号系统。SSH 断开后后台服务和训练仍然运行。服务器重启后需重新启动工作台；未可靠完成的运行不会被自动当成成功。

TensorBoard 使用独立的本机空闲端口。工作台启动后查看返回 URL 中的端口 P，再添加 `ssh -L P:127.0.0.1:P ...`；浏览器可通过该端口访问。工作台页面的 8765 转发不会自动转发其他端口。

## 连接其他训练项目

“本机设置 → 添加项目”创建独立档案，修改项目目录、Python、训练入口、任务 ID 和注册模块后保存。多个项目共用一个串行调度器，不会并行启动受管理训练。

项目档案示例：

```json
{
  "id": "my-go2",
  "name": "My Go2 Project",
  "root": "/data/my_project",
  "python": "/opt/conda/envs/isaaclab/bin/python",
  "isaac_sim": "",
  "environment": {},
  "train_entry": "scripts/rsl_rl/train.py",
  "task": "AME-Go2-v0",
  "imports": ["ame_locomotion.tasks"],
  "adapter": "isaaclab_rsl_rl"
}
```

当前适配器要求：

- 训练入口使用 Isaac Lab 标准任务注册、argparse + Hydra，支持已提供的训练参数、`--experiment_name` 和 `--logger tensorboard`。
- 任务注册提供 `env_cfg_entry_point`、`rsl_rl_cfg_entry_point`。配置导出读取 `env.rewards.<term>.weight`；存在 `reward_gates` 时导出对应标量，没有则不展示门控项。
- resume 支持绝对 `--checkpoint`；仅加载权重需要入口提供 `--finetune_from`。普通 Isaac Lab 项目可能没有这个参数；这时选择从头训练，或扩展适配器映射该项目自己的加载方式。
- 使用 RSL-RL 的 `model_<编号>.pt` 格式，包含模型状态和迭代计数；真实最终编号遵循当前已验证 runner 的保存语义。

不同 CLI 或 checkpoint 格式的项目，需要扩展 `backend/workbench/adapters.py` 中命令构建、配置校验、日志解析，以及 `adapters/export_config.py` / `checkpoint_info.py`。先用独立模拟夹具测试，再用真实短训练验证。不能仅修改任务名就保证任意训练项目兼容。

## 环境与存储

- Conda Python 会设置其自身目录、Scripts/Library bin（Windows）或 bin（Linux）到子进程 PATH，不执行任意 shell 激活脚本。
- 独立 Isaac Sim 目录会生成 `ISAAC_PATH`、`CARB_APP_PATH`、`EXP_PATH` 和 site 路径。其他特定变量填写在项目的 environment 中。
- 不保存 SSH 登录凭据。当前方案是在训练所在机器运行工作台，通过 SSH 转发使用网页。
- 项目档案、命令、配置和运行记录会保存环境变量，因此不要在项目表单中填写密钥。
- 更换机器建议重新导入实验计划；旧运行记录含原机器的绝对路径和进程身份，不作为跨机器自动恢复依据。历史 checkpoint 与日志可单独复制归档。
- 新实验的原生训练结果保存在目标项目的 `logs/rsl_rl/<任务目录>/<时间戳>_<实验名>/`，任务目录来自原配置的 `experiment_name`。迁移 checkpoint 和事件文件时请同时考虑训练项目的 logs 目录；旧版本在工作台 outputs 或 ID 目录中保存的结果仍可浏览和接续。
