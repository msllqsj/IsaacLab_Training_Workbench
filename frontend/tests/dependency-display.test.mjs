import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { buildSync } from 'esbuild'
import { dependencyEdges } from '../src/queueLayout.ts'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

const require=createRequire(import.meta.url)
const output=buildSync({entryPoints:[fileURLToPath(new URL('../src/QueueTable.tsx',import.meta.url))],bundle:true,write:false,platform:'node',format:'cjs',packages:'external',jsx:'automatic'}).outputFiles[0].text
const module={exports:{}}
new Function('require','module','exports',output)(require,module,module.exports)
const {QueueTable}=module.exports
const job=(id,status,checkpoint={kind:'job',job_id:'root',path:''})=>({id,position:0,status,config:{training:{run_name:id,seed:42,max_iterations:2},mode:'new',checkpoint}})
const render=jobs=>renderToStaticMarkup(React.createElement(QueueTable,{jobs,runs:[],selectedId:'',actions:{},badge:status=>status}))
const cells=html=>Array.from(html.matchAll(/<div\b[^>]*class="dependency-value ([^"]*)"[^>]*>(.*?)<\/div>/g),match=>({state:match[1],content:match[2]}))

test('running and completed dependencies are solid while other states remain hollow',()=>{
  const statuses=['succeeded','queued','draft','starting','running','stopping','failed','cancelled','skipped','blocked','interrupted']
  const root=job('root','succeeded',{kind:'none',job_id:'',path:''})
  const result=cells(render([root,...statuses.map(status=>job(status,status))]))
  assert.equal(result.length,statuses.length+1)
  assert.doesNotMatch(result[0].content,/dependency-node/)
  assert.match(result[0].content,/>—<\/span>/)
  statuses.forEach((status,index)=>{
    assert.equal(result[index+1].state==='active',status==='running',status)
    assert.equal(result[index+1].state==='complete',status==='succeeded',status)
    assert.match(result[index+1].content,/dependency-node connected/)
    assert.match(result[index+1].content,/>01<\/span>/)
    assert.doesNotMatch(result[index+1].content,/>—<\/span>/)
  })
})

test('no upstream job shows a dash; missing dependencies show an error instead of a dash',()=>{
  const result=cells(render([
    job('file','draft',{kind:'file',job_id:'',path:'/models/model_1.pt'}),
    job('missing','queued'),
  ]))
  assert.doesNotMatch(result[0].content,/dependency-node/)
  assert.match(result[0].content,/>—<\/span>/)
  assert.equal(result[1].state,'missing')
  assert.match(result[1].content,/>失效<\/span>/)
  assert.doesNotMatch(result[1].content,/>—<\/span>/)
})

test('root and branch sources retain line anchors even when they show a dash without a dot',()=>{
  const jobs=[job('root','succeeded',{kind:'none',job_id:'',path:''}),job('child','succeeded'),job('branch','queued')]
  const html=render(jobs)
  const anchors=new Set(Array.from(html.matchAll(/data-dependency-anchor="([^"]+)"/g),match=>match[1]))
  assert.equal(anchors.size,jobs.length)
  for(const edge of dependencyEdges(jobs)){
    assert.ok(anchors.has(edge.from),edge.from)
    assert.ok(anchors.has(edge.to),edge.to)
  }
  const root=cells(html)[0]
  assert.match(root.content,/>—<\/span>/)
  assert.doesNotMatch(root.content,/dependency-node/)
})
