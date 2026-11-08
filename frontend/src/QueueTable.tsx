import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowDown, ArrowUp, Copy, Edit3, FileText, FolderOpen, RefreshCw, SkipForward, Trash2 } from 'lucide-react'
import { ACTIVE, type Job, type Run } from './types'
import { canMoveJob, dependencyEdges, dependencyPath, menuPosition, type Point } from './queueLayout'

type Actions={
  select:(id:string)=>void
  copy:(job:Job)=>void
  edit:(job:Job)=>void
  move:(job:Job,delta:number)=>void
  retry:(job:Job)=>void
  skip:(job:Job)=>void
  details:(run:Run)=>void
  remove:(job:Job)=>void
}
type Menu={jobId:string;point:Point;anchor:HTMLTableRowElement}
const number=(value:number)=>value.toLocaleString('en-US')
const sequence=(job:Job)=>String(job.position+1).padStart(2,'0')
const filename=(path:string)=>path.split(/[\\/]/).at(-1)??path

function RowContextMenu({menu,job,jobs,run,actions,close}:{menu:Menu;job:Job;jobs:Job[];run?:Run;actions:Actions;close:(restoreFocus?:boolean)=>void}){
  const ref=useRef<HTMLDivElement>(null)
  const [point,setPoint]=useState(menu.point)
  useLayoutEffect(()=>{
    const element=ref.current
    if(!element)return
    setPoint(menuPosition(menu.point,element.getBoundingClientRect(),{width:window.innerWidth,height:window.innerHeight}))
    element.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus({preventScroll:true})
  },[menu])
  useEffect(()=>{
    const outside=(event:PointerEvent)=>{if(!ref.current?.contains(event.target as Node))close()}
    const key=(event:KeyboardEvent)=>{
      if(event.key==='Escape'){event.preventDefault();close(true)}
      else if(event.key==='Tab')close()
    }
    const dismiss=(event:Event)=>{if(!ref.current?.contains(event.target as Node))close()}
    document.addEventListener('pointerdown',outside,true)
    document.addEventListener('keydown',key,true)
    window.addEventListener('scroll',dismiss,true)
    window.addEventListener('resize',dismiss)
    window.addEventListener('blur',dismiss)
    return()=>{
      document.removeEventListener('pointerdown',outside,true)
      document.removeEventListener('keydown',key,true)
      window.removeEventListener('scroll',dismiss,true)
      window.removeEventListener('resize',dismiss)
      window.removeEventListener('blur',dismiss)
    }
  },[close])
  const choose=(action:()=>void)=>{close();action()}
  const moveFocus=(key:string)=>{
    const buttons=Array.from(ref.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')??[])
    if(!buttons.length)return
    const current=buttons.indexOf(document.activeElement as HTMLButtonElement)
    const index=key==='Home'?0:key==='End'?buttons.length-1:(current+(key==='ArrowDown'?1:-1)+buttons.length)%buttons.length
    buttons[index].focus()
  }
  return createPortal(<div ref={ref} className="experiment-context-menu" role="menu" aria-label={`${job.config.training.run_name} 实验操作`}
    style={{left:point.x,top:point.y}} onContextMenu={event=>event.preventDefault()}
    onKeyDown={event=>{if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){event.preventDefault();moveFocus(event.key)}}}>
    <div className="context-menu-title">实验 {sequence(job)}<span>{job.config.training.run_name}</span></div>
    <button role="menuitem" onClick={()=>choose(()=>actions.copy(job))}><Copy size={18}/>复制为新实验</button>
    {['draft','blocked','queued'].includes(job.status)&&!run&&!job.attempt_count&&<button role="menuitem" onClick={()=>choose(()=>actions.edit(job))}><Edit3 size={18}/>编辑配置</button>}
    {run&&<button role="menuitem" onClick={()=>choose(()=>actions.details(run))}><FileText size={18}/>查看运行详情</button>}
    <div className="context-menu-separator" role="separator"/>
    <button role="menuitem" disabled={!canMoveJob(jobs,job.id,-1)} onClick={()=>choose(()=>actions.move(job,-1))}><ArrowUp size={18}/>向前移动</button>
    <button role="menuitem" disabled={!canMoveJob(jobs,job.id,1)} onClick={()=>choose(()=>actions.move(job,1))}><ArrowDown size={18}/>向后移动</button>
    {['failed','interrupted','cancelled','blocked'].includes(job.status)&&<button role="menuitem" onClick={()=>choose(()=>actions.retry(job))}><RefreshCw size={18}/>重新尝试</button>}
    {!ACTIVE.includes(job.status)&&job.status!=='succeeded'&&<><div className="context-menu-separator" role="separator"/><button role="menuitem" onClick={()=>choose(()=>actions.skip(job))}><SkipForward size={18}/>跳过实验</button></>}
    <div className="context-menu-separator" role="separator"/>
    <button role="menuitem" className="danger-text" disabled={ACTIVE.includes(job.status)||jobs.some(other=>other.config.checkpoint.kind==='job'&&other.config.checkpoint.job_id===job.id)} title={jobs.some(other=>other.config.checkpoint.kind==='job'&&other.config.checkpoint.job_id===job.id)?'仍被其他实验依赖，请先修改来源或删除依赖项':ACTIVE.includes(job.status)?'请先停止训练':'从队列中删除实验，保留训练文件'} onClick={()=>choose(()=>actions.remove(job))}><Trash2 size={18}/>删除实验</button>
  </div>,document.body)
}

export function QueueTable({jobs,runs,selectedId,actions,badge}:{jobs:Job[];runs:Run[];selectedId:string;actions:Actions;badge:(status:string)=>ReactNode}){
  const frame=useRef<HTMLDivElement>(null)
  const anchors=useRef(new Map<string,HTMLDivElement>())
  const [geometry,setGeometry]=useState<{width:number;height:number;points:Record<string,Point>}>({width:0,height:0,points:{}})
  const [menu,setMenu]=useState<Menu|null>(null)
  const edges=dependencyEdges(jobs)
  const outgoing=new Set(edges.map(edge=>edge.from))
  const layoutKey=jobs.map(job=>`${job.id}:${job.status}:${job.config.checkpoint.job_id}`).join('|')
  useLayoutEffect(()=>{
    const element=frame.current
    if(!element)return
    const measure=()=>{
      const bounds=element.getBoundingClientRect()
      const points:Record<string,Point>={}
      for(const [id,node] of anchors.current){
        const marker=node.querySelector<HTMLElement>('.dependency-node, .dependency-label')
        if(!marker)continue
        const rect=marker.getBoundingClientRect()
        points[id]={x:rect.x-bounds.x+rect.width/2,y:rect.y-bounds.y+rect.height/2}
      }
      setGeometry({width:bounds.width,height:bounds.height,points})
    }
    measure()
    const observer=new ResizeObserver(measure);observer.observe(element)
    return()=>observer.disconnect()
  },[layoutKey])
  useEffect(()=>setMenu(null),[layoutKey])
  const close=(restoreFocus=false)=>{if(restoreFocus)menu?.anchor.focus({preventScroll:true});setMenu(null)}
  const menuJob=menu?jobs.find(job=>job.id===menu.jobId):undefined
  return <div className="table-scroll"><div ref={frame} className="queue-table-frame">
    <table className="experiment-table"><thead><tr><th className="number-col">#</th><th>实验名称</th><th>状态</th><th>新增迭代</th><th className="checkpoint-heading">检查点</th><th className="dependency-heading">依赖</th></tr></thead>
      <tbody>{jobs.map(job=>{
        const run=runs.find(run=>run.job_id===job.id),cp=job.config.checkpoint
        const parent=cp.kind==='job'?jobs.find(candidate=>candidate.id===cp.job_id):undefined
        const input=cp.kind==='file'?cp.path:cp.kind==='job'?runs.find(run=>run.job_id===cp.job_id&&run.status==='succeeded')?.data.final_checkpoint??parent?.last_result?.final_checkpoint:undefined
        const checkpoint=run?.data.final_checkpoint??job.last_result?.final_checkpoint??run?.data.checkpoint??job.last_result?.checkpoint??input
        const progress=run?.progress??job.last_result?.progress
        const missing=cp.kind==='job'&&!parent
        const dependencyTitle=parent?`实验 ${sequence(job)} 依赖实验 ${sequence(parent)} · ${parent.config.training.run_name}`:missing?'上游实验不存在':outgoing.has(job.id)?'此实验为后续实验提供 checkpoint':'无上游实验'
        return <tr key={job.id} tabIndex={0} aria-selected={job.id===selectedId}
          className={`${job.id===selectedId?'selected':''} ${ACTIVE.includes(job.status)?'is-running':''}`}
          onClick={()=>actions.select(job.id)}
          onContextMenu={event=>{event.preventDefault();actions.select(job.id);setMenu({jobId:job.id,point:{x:event.clientX,y:event.clientY},anchor:event.currentTarget})}}
          onKeyDown={event=>{
            if(event.key==='ContextMenu'||(event.shiftKey&&event.key==='F10')){
              event.preventDefault();actions.select(job.id);const rect=event.currentTarget.getBoundingClientRect()
              setMenu({jobId:job.id,point:{x:rect.left+110,y:rect.top+rect.height/2},anchor:event.currentTarget})
            }else if(event.key==='Enter')actions.select(job.id)
          }}>
          <td className="number-col"><span className="row-index">{sequence(job)}</span></td>
          <td><strong className="experiment-name" title={job.config.training.run_name}>{job.config.training.run_name}</strong><small className="row-subtitle">{job.config.mode==='new'?'从头训练':job.config.mode==='resume'?'恢复训练':'仅加载权重'} · seed {job.config.training.seed}</small></td>
          <td>{badge(job.status)}</td>
          <td><div className="iteration-cell"><span>{number(progress?.completed_updates??0)} <span className="muted">/ {number(job.config.training.max_iterations)}</span></span>{ACTIVE.includes(job.status)&&<div className="progress-row"><div className="progress-track"><i style={{width:`${(progress?.progress??0)*100}%`}}/></div><small>{Math.round((progress?.progress??0)*100)}%</small></div>}</div></td>
          <td className="checkpoint-cell">{checkpoint?<span title={`${(run?.data.final_checkpoint??job.last_result?.final_checkpoint)?'最终检查点':'输入检查点'}：${checkpoint}`}><FolderOpen size={19}/><span>{filename(checkpoint)}</span></span>:<span className="muted">{cp.kind==='job'?'待上游完成':ACTIVE.includes(job.status)?'生成中':'—'}</span>}</td>
          <td className="dependency-cell"><div ref={node=>{if(node)anchors.current.set(job.id,node);else anchors.current.delete(job.id)}} data-dependency-anchor={job.id} className={`dependency-value ${missing?'missing':job.status==='running'?'active':job.status==='succeeded'?'complete':''}`} title={dependencyTitle}>
            {cp.kind==='job'&&<span className="dependency-node connected" aria-hidden="true"/>}
            <span className="dependency-label">{parent?sequence(parent):missing?'失效':'—'}</span><span className="sr-only">{dependencyTitle}</span>
          </div></td>
        </tr>
      })}</tbody>
    </table>
    <svg className="dependency-lines" width={geometry.width} height={geometry.height} aria-hidden="true">
      {edges.map(edge=>{
        const from=geometry.points[edge.from],to=geometry.points[edge.to]
        if(!from||!to)return null
        return <path key={edge.to} d={dependencyPath(edge,from,to)}/>
      })}
    </svg>
  </div>{menu&&menuJob&&<RowContextMenu menu={menu} job={menuJob} jobs={jobs} run={runs.find(run=>run.job_id===menuJob.id)} actions={actions} close={close}/>}</div>
}
