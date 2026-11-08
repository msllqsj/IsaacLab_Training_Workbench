import { useEffect, useState } from 'react'
import { Save } from 'lucide-react'
import { Button, Busy } from './components'
import { api } from './types'

type Note = {text:string;updated:number|null}

export function NoteField({text,disabled,onChange}:{text:string;disabled:boolean;onChange:(text:string)=>void}){
  return <label className="form-field"><span>笔记 / 备注</span><textarea rows={12} value={text} disabled={disabled} maxLength={20000} placeholder="记录实验目的、参数调整、观察结果及后续想法…" onChange={event=>onChange(event.target.value)}/></label>
}

export function RunNotes({runId,onDirtyChange}:{runId:string;onDirtyChange:(dirty:boolean)=>void}){
  const [saved,setSaved]=useState<Note|null>(null),[text,setText]=useState(''),[saving,setSaving]=useState(false),[error,setError]=useState(''),[reload,setReload]=useState(0)
  const dirty=saved!==null&&text!==saved.text
  useEffect(()=>{
    let cancelled=false
    void api<Note>(`/runs/${runId}/note`).then(note=>{if(!cancelled){setSaved(note);setText(note.text);setError('')}}).catch(error=>{if(!cancelled)setError(String(error))})
    return()=>{cancelled=true}
  },[runId,reload])
  useEffect(()=>{onDirtyChange(dirty)},[dirty,onDirtyChange])
  useEffect(()=>{
    if(!dirty)return
    const warn=(event:BeforeUnloadEvent)=>{event.preventDefault();event.returnValue=''}
    window.addEventListener('beforeunload',warn)
    return()=>window.removeEventListener('beforeunload',warn)
  },[dirty])
  async function save(){
    setSaving(true);setError('')
    try{setSaved(await api<Note>(`/runs/${runId}/note`,'PUT',{text}))}
    catch(error){setError(error instanceof Error?error.message:String(error))}
    finally{setSaving(false)}
  }
  return <section className="run-notes" aria-label="实验笔记">
    <NoteField text={text} disabled={!saved||saving} onChange={setText}/>
    <div className="run-notes-actions"><span className="muted" role="status">{dirty?'有未保存的修改':saved?.updated?`已保存 · ${new Date(saved.updated*1000).toLocaleString('zh-CN',{hour12:false})}`:saved?'尚未添加笔记':'正在读取笔记…'}</span><Button variant="primary" disabled={!dirty||saving} onClick={()=>void save()}>{saving?<Busy>保存中…</Busy>:<><Save size={16}/>保存笔记</>}</Button></div>
    <p className="muted small">与同一实验的编辑配置共享笔记，最多 20,000 字；编辑后请点击保存。</p>
    {error&&<div className="error-box" role="alert">{error}{!saved&&<Button onClick={()=>setReload(value=>value+1)}>重试读取</Button>}</div>}
  </section>
}
