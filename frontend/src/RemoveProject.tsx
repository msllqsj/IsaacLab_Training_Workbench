import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import { Button, Busy, Modal } from './components'
import type { Project } from './types'

export function RemoveProject({project,disabled,onRemove}:{project:Project;disabled:boolean;onRemove:()=>Promise<void>}){
  const [open,setOpen]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState('')
  async function confirm(){try{setBusy(true);setError('');await onRemove();setOpen(false)}catch(error){setError(error instanceof Error?error.message:String(error))}finally{setBusy(false)}}
  return <><Button variant="ghost" className="danger-text" disabled={disabled} title={disabled?'请先保存新连接或停止该项目的训练':'从工作台移除此项目，保留本地文件'} onClick={()=>setOpen(true)}><Trash2 size={16}/>删除项目连接</Button>
    {open&&<Modal title="移除项目连接？" description="此项目及关联的工作流、实验记录将从工作台中移除。" onClose={()=>{if(!busy)setOpen(false)}}>
      <div className="editor-body delete-dialog-body"><strong>{project.name}</strong><code className="delete-retained-path">{project.root}</code><p>本地项目代码、模型、checkpoint 和日志文件全部保留。</p>{error&&<div className="error-box" role="alert">{error}</div>}</div>
      <div className="modal-footer"><Button disabled={busy} onClick={()=>setOpen(false)}>取消</Button><Button variant="danger" disabled={busy} onClick={()=>void confirm()}>{busy?<Busy>移除中…</Busy>:<><Trash2 size={16}/>确认移除连接</>}</Button></div>
    </Modal>}</>
}
