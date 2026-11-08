import { useState, type ReactNode } from 'react'
import { ChevronRight, FileText, Trash2 } from 'lucide-react'
import { Button, Busy, Empty, Modal } from './components'
import { ACTIVE, type Run } from './types'

export type DeleteTarget={kind:'run'|'job';id:string;name:string;attempt?:number;directory?:string}

export function HistoryList({runs,badge,onOpen,onDelete}:{runs:Run[];badge:(status:string)=>ReactNode;onOpen:(run:Run)=>void;onDelete:(run:Run)=>void}){
  if(!runs.length)return <Empty icon={<FileText size={34}/>}>暂无匹配的运行记录</Empty>
  return <div role="list">{runs.map(run=><div className="history-row" role="listitem" key={run.id}>
    <button className="history-open" onClick={()=>onOpen(run)} aria-label={`查看 ${run.data.config.training.run_name} 第 ${run.number} 次运行`}>
      <div className="history-icon"><FileText size={21}/></div>
      <div className="history-name"><strong>{run.data.config.training.run_name}</strong><small>{new Date(run.created*1000).toLocaleString('zh-CN',{hour12:false})} · 第 {run.number} 次尝试</small></div>
      {badge(run.status)}
      <span className="history-progress">{(run.progress.completed_updates??0).toLocaleString('en-US')} / {run.data.config.training.max_iterations.toLocaleString('en-US')}</span>
      <ChevronRight size={18}/>
    </button>
    <Button variant="ghost" className="history-delete danger-text" disabled={ACTIVE.includes(run.status)} title={ACTIVE.includes(run.status)?'请等待训练结束后删除':'删除记录，保留模型和日志文件'} aria-label={`删除 ${run.data.config.training.run_name} 第 ${run.number} 次运行记录`} onClick={()=>onDelete(run)}><Trash2 size={17}/><span>删除</span></Button>
  </div>)}</div>
}

export function DeleteDialog({target,onClose,onDelete}:{target:DeleteTarget;onClose:()=>void;onDelete:(deleteFiles:boolean)=>Promise<void>}){
  const [busy,setBusy]=useState(false),[error,setError]=useState('')
  const [deleteFiles,setDeleteFiles]=useState(false)
  async function confirm(){try{setBusy(true);setError('');await onDelete(deleteFiles)}catch(error){setError(error instanceof Error?error.message:String(error));setBusy(false)}}
  const experiment=target.kind==='job'
  return <Modal title={experiment?'删除实验？':'删除运行记录？'} description={experiment?'将从队列移除此实验及其运行记录。':'将从实验记录列表移除此项运行。'} onClose={()=>{if(!busy)onClose()}}>
    <div className="editor-body delete-dialog-body"><strong>{target.name}</strong>{target.attempt!==undefined&&<span className="muted">第 {target.attempt} 次尝试</span>}
      <label className="delete-file-option"><input type="checkbox" checked={deleteFiles} disabled={busy} onChange={event=>setDeleteFiles(event.target.checked)}/>同时删除本次训练文件，释放磁盘空间</label>
      <p>{deleteFiles?'将删除本次目录内的模型、checkpoint、视频、TensorBoard 数据及文字日志，无法撤销。被其他实验依赖的文件会受到保护。':'仅删除列表记录，模型和日志文件保留；队列中的结果和 checkpoint 接续仍可使用。'}</p>
      {target.directory&&<code className="delete-retained-path">{target.directory}</code>}
      {error&&<div className="error-box" role="alert">{error}</div>}
    </div>
    <div className="modal-footer"><Button disabled={busy} onClick={onClose}>取消</Button><Button variant="danger" disabled={busy} onClick={()=>void confirm()}>{busy?<Busy>删除中…</Busy>:<><Trash2 size={17}/>{experiment?'删除实验':'删除记录'}</>}</Button></div>
  </Modal>
}
