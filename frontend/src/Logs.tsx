import { useEffect, useRef, useState } from 'react'
import { ArrowDown, Copy, ExternalLink, Pause, Play, Terminal, Trash2 } from 'lucide-react'
import { Button, Busy, Modal } from './components'
import { api } from './types'
import { latestRequest } from './latestRequest'

type LogPage={text:string;offset:number;generation:string;reset:boolean}

export function Logs({runId,compact=false}:{runId?:string;compact?:boolean}){
  const [text,setText]=useState(''),[paused,setPaused]=useState(false),[auto,setAuto]=useState(true)
  const [confirm,setConfirm]=useState(false),[clearing,setClearing]=useState(false),[error,setError]=useState(''),[reload,setReload]=useState(0)
  const [tbStarting,setTbStarting]=useState(false),tbRequest=useRef(false)
  const offset=useRef(0),first=useRef(true),generation=useRef(''),pre=useRef<HTMLPreElement>(null)
  const requests=useRef(latestRequest<LogPage>(page=>{
    first.current=false;offset.current=page.offset;generation.current=page.generation
    setText(old=>(page.reset?page.text:old+page.text).slice(-80000))
  }))
  useEffect(()=>{requests.current.invalidate();setText('');offset.current=0;first.current=true;generation.current='';setError('')},[runId])
  useEffect(()=>{
    if(!runId||paused||clearing)return
    const poll=()=>void requests.current.run(()=>api<LogPage>(`/runs/${runId}/logs?offset=${offset.current}&tail=${first.current}&generation=${encodeURIComponent(generation.current)}`)).catch(error=>setError(String(error)))
    poll();const timer=setInterval(poll,1200)
    return()=>{clearInterval(timer);requests.current.invalidate()}
  },[runId,paused,clearing,reload])
  useEffect(()=>{if(auto&&pre.current)pre.current.scrollTop=pre.current.scrollHeight},[text,auto])
  async function clear(){
    if(!runId)return
    requests.current.invalidate();setClearing(true);setError('')
    try{
      const page=await api<LogPage>(`/runs/${runId}/logs`,'DELETE')
      requests.current.invalidate();offset.current=page.offset;generation.current=page.generation;first.current=false
      setText('');setConfirm(false);setReload(value=>value+1)
    }catch(error){setError(error instanceof Error?error.message:String(error))}
    finally{setClearing(false)}
  }
  async function tensorboard(){
    if(!runId||tbRequest.current)return
    tbRequest.current=true;setTbStarting(true);setError('')
    let tab:Window|null=null
    try{
      tab=window.open('about:blank','_blank')
      if(!tab)throw new Error('浏览器阻止了新窗口，请允许本站弹出窗口后重试。')
      tab.document.title='正在启动 TensorBoard'
      const body=tab.document.body
      if(body){
        body.textContent='正在启动 TensorBoard，请稍候，服务就绪后会自动打开。'
        body.style.cssText='font:16px system-ui,sans-serif;padding:48px;line-height:1.8'
      }
      const result=await api<{url:string}>(`/runs/${runId}/tensorboard`,'POST')
      if(!tab.closed)tab.location.replace(result.url)
    }catch(error){tab?.close();setError(error instanceof Error?error.message:String(error))}
    finally{tbRequest.current=false;setTbStarting(false)}
  }

  return <><section className={`panel logs-panel ${compact?'compact':''}`}><div className="panel-heading"><h2><Terminal size={18}/>运行日志</h2><div className="actions"><span className="muted small">文字日志保留最近 10 MB</span><Button className="small-button" disabled={!runId||tbStarting} onClick={()=>void tensorboard()}>{tbStarting?<Busy>正在启动…</Busy>:<>TensorBoard<ExternalLink size={14}/></>}</Button><button className="icon-button" title="复制日志" disabled={!text} onClick={()=>void navigator.clipboard.writeText(text)}><Copy size={16}/></button><button className={`icon-button ${paused?'selected':''}`} title={paused?'继续接收日志':'暂停接收日志'} disabled={!runId} onClick={()=>setPaused(!paused)}>{paused?<Play size={16}/>:<Pause size={16}/>}</button><button className={`icon-button ${auto?'selected':''}`} title="自动滚动" disabled={!runId} onClick={()=>setAuto(!auto)}><ArrowDown size={16}/></button><button className="icon-button" title="清空已保存的文字日志" disabled={!runId||clearing} onClick={()=>setConfirm(true)}><Trash2 size={16}/></button></div></div>{error&&!confirm&&<div className="error-box">{error}</div>}<pre ref={pre} className="log-content">{text||<span className="log-placeholder">{runId?'暂无日志；新输出会继续显示。':'选择实验后将在这里显示实时训练日志。'}</span>}</pre></section>
    {confirm&&<Modal title="清空文字日志？" description="将清空此运行磁盘上已保存的 console.log，刷新页面后也不会恢复。" onClose={()=>{if(!clearing)setConfirm(false)}}><div className="editor-body"><p>模型、checkpoint 和 TensorBoard 数据保留。正在运行的训练会继续，新日志仍会追加。</p>{error&&<div className="error-box" role="alert">{error}</div>}</div><div className="modal-footer"><Button disabled={clearing} onClick={()=>setConfirm(false)}>取消</Button><Button variant="danger" disabled={clearing} onClick={()=>void clear()}>{clearing?<Busy>清空中…</Busy>:<><Trash2 size={16}/>清空文字日志</>}</Button></div></Modal>}</>
}
