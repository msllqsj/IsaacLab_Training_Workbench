import math
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def uid():
    return uuid.uuid4().hex


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=uid)
    name: str = "AME Locomotion · Go2"
    root: str
    python: str
    isaac_sim: str = ""
    environment: dict[str, str] = Field(default_factory=dict)
    train_entry: str = "scripts/rsl_rl/train.py"
    task: str = "AME-Go2-v0"
    imports: list[str] = ["ame_locomotion.tasks"]
    adapter: Literal["isaaclab_rsl_rl", "mock"] = "isaaclab_rsl_rl"

    @field_validator("id")
    @classmethod
    def valid_id(cls, v):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", v):
            raise ValueError("Invalid project id")
        return v


class Training(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str = "experiment"
    max_iterations: int = Field(default=2000, ge=1)
    num_envs: int = Field(default=1024, ge=1)
    seed: int = Field(default=42, ge=0)
    device: str = "cuda:0"
    save_interval: int = Field(default=100, ge=1)
    headless: bool = True
    video: bool = False
    video_length: int = Field(default=200, ge=1)
    video_interval: int = Field(default=24000, ge=1)

    @field_validator("run_name")
    @classmethod
    def valid_name(cls, v):
        if not re.fullmatch(r"[\w.-]{1,100}", v, re.UNICODE) or v in (".", ".."):
            raise ValueError("名称只能使用字母、数字、中文、下划线、点和短横线")
        return v

    @field_validator("device")
    @classmethod
    def valid_device(cls, v):
        if not re.fullmatch(r"cuda(?::\d+)?|cpu", v):
            raise ValueError("设备应为 cuda:0 或 cpu")
        return v


class CheckpointSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["none", "file", "job"] = "none"
    path: str = ""
    job_id: str = ""


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training: Training = Field(default_factory=Training)
    mode: Literal["new", "resume", "warm_start"] = "new"
    checkpoint: CheckpointSource = Field(default_factory=CheckpointSource)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_mode(self):
        if self.mode == "new" and self.checkpoint.kind != "none":
            raise ValueError("从头训练不能指定 checkpoint")
        if self.mode != "new" and self.checkpoint.kind == "none":
            raise ValueError("接续训练需要选择 checkpoint")
        if self.checkpoint.kind == "file" and not self.checkpoint.path.strip():
            raise ValueError("checkpoint 路径不能为空")
        if self.checkpoint.kind == "job" and not self.checkpoint.job_id:
            raise ValueError("请选择上游实验")
        for key, v in self.overrides.items():
            if not re.fullmatch(r"env\.(?:rewards\.[A-Za-z0-9_]+\.weight|reward_gates\.[A-Za-z0-9_.]+)", key):
                raise ValueError(f"不支持的配置字段: {key}")
            if v is not None and not isinstance(v, (bool, int, float)):
                raise ValueError(f"配置值必须为数值、布尔或 null: {key}")
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("配置值必须为有限数值")
        return self


class ExperimentSave(Experiment):
    note: str = Field(default="", max_length=20000)


class WorkflowCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    project_id: str


class GridRequest(BaseModel):
    base: Experiment
    axes: dict[str, list[Any]]
