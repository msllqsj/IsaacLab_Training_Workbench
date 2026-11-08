import type { Job } from './types'

export type DependencyEdge = {from:string;to:string;fromIndex:number;toIndex:number;lane:number}
export type Point = {x:number;y:number}

export function dependencyEdges(jobs:Job[]):DependencyEdge[]{
  const indices=new Map(jobs.map((job,index)=>[job.id,index]))
  const lanes:number[]=[]
  const edges:DependencyEdge[]=[]
  jobs.forEach((job,toIndex)=>{
    const cp=job.config.checkpoint
    if(cp.kind!=='job')return
    const fromIndex=indices.get(cp.job_id)
    if(fromIndex===undefined||fromIndex>=toIndex)return
    let lane=0
    if(toIndex-fromIndex>1){
      lane=lanes.findIndex(end=>end<=fromIndex)
      if(lane<0)lane=lanes.length
      lanes[lane]=toIndex
      lane+=1
    }
    edges.push({from:cp.job_id,to:job.id,fromIndex,toIndex,lane})
  })
  return edges
}

export function dependencyPath(edge:DependencyEdge,from:Point,to:Point):string{
  const start=from.y+9,end=to.y-9
  if(edge.toIndex-edge.fromIndex===1)return `M ${from.x} ${start} L ${to.x} ${end}`
  const laneX=Math.min(from.x,to.x)-Math.min(38,20+edge.lane*6)
  return `M ${from.x} ${start} C ${laneX} ${start+12}, ${laneX} ${end-12}, ${to.x} ${end}`
}

export function menuPosition(point:Point,size:{width:number;height:number},viewport:{width:number;height:number}):Point{
  const margin=10
  return {
    x:Math.max(margin,Math.min(point.x,viewport.width-size.width-margin)),
    y:Math.max(margin,Math.min(point.y,viewport.height-size.height-margin)),
  }
}

export function canMoveJob(jobs:Job[],id:string,delta:number):boolean{
  if(jobs.some(job=>['starting','running','stopping'].includes(job.status)))return false
  const ids=jobs.map(job=>job.id),index=ids.indexOf(id),target=index+delta
  if(index<0||target<0||target>=ids.length)return false
  ;[ids[index],ids[target]]=[ids[target],ids[index]]
  const order=new Map(ids.map((jid,i)=>[jid,i]))
  return jobs.every(job=>job.config.checkpoint.kind!=='job'||
    (order.get(job.config.checkpoint.job_id)??Infinity)<order.get(job.id)!)
}
