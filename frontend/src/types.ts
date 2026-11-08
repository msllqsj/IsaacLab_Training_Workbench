export type Training = {run_name:string;max_iterations:number;num_envs:number;seed:number;device:string;save_interval:number;headless:boolean;video:boolean;video_length:number;video_interval:number}
export type Spec = {training:Training;mode:'new'|'resume'|'warm_start';checkpoint:{kind:'none'|'file'|'job';path:string;job_id:string};overrides:Record<string,number|boolean|null>}
export type Project = {id:string;name:string;root:string;python:string;isaac_sim?:string;environment?:Record<string,string>;train_entry:string;task:string;imports:string[];adapter:'isaaclab_rsl_rl'|'mock'}
export type Workflow = {id:string;name:string;project_id:string;created:number}
export type Job = {id:string;workflow_id:string;position:number;config:Spec;baseline:{hash:string};status:string;message:string;updated:number;attempt_count?:number;last_result?:{status:string;progress:Progress;final_checkpoint?:string;checkpoint?:string}}
export type Progress = {phase?:string;progress?:number;iteration?:number;start_iteration?:number;total_iteration?:number;completed_updates?:number;loaded_iteration?:number;mean_reward?:number;eta?:string;started?:number}
export type Run = {id:string;job_id:string;number:number;status:string;created:number;updated:number;progress:Progress;command_display:string;data:{directory:string;output_root:string;command:string[];project:Project;config:Spec;checkpoint?:string;message?:string;run_dir?:string;final_checkpoint?:string;iteration?:number;finished?:number}}
export type Queue = {enabled:boolean;workflow_id:string|null;message:string;wait_for?:{pid:number;created:number}|null}
export type Snapshot = {projects:Project[];workflows:Workflow[];jobs:Record<string,Job[]>;runs:Run[];queue:Queue;worker_alive:boolean}
export type Field = {path:string;label:string;value:number|boolean|null;kind:'bool'|'int'|'float'|'none';group:'rewards'|'gates';nullable:boolean}
export type Schema = {stage:number|null;fields:Field[];training:Training;versions:Record<string,string>;stale:boolean;exported_at:number;fingerprint:{hash:string}}
export type SchemaResponse = {schema:Schema|null;probe:{status:string;message:string}}

export async function api<T>(path:string, method='GET', body?:unknown):Promise<T>{
  const response=await fetch('/api'+path,{method,cache:'no-store',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)})
  const data=await response.json()
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail??data))
  return data as T
}
export const ACTIVE=['starting','running','stopping']
export const statusLabels:Record<string,string>={draft:'草稿',queued:'等待中',blocked:'依赖阻塞',starting:'初始化',running:'运行中',stopping:'停止中',succeeded:'已完成',failed:'失败',cancelled:'已取消',skipped:'已跳过',interrupted:'已中断'}
