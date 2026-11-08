import { test } from 'node:test'
import assert from 'node:assert/strict'
import { canMoveJob, dependencyEdges, dependencyPath, menuPosition } from '../src/queueLayout.ts'

const job=(id,parent,kind=parent?'job':'none')=>({id,position:Number(id)-1,status:'draft',config:{checkpoint:{kind,job_id:parent??'',path:''}}})

test('dependency edges preserve roots, non-adjacent references and branches',()=>{
  const edges=dependencyEdges([job('1'),job('2','1'),job('3','1'),job('4','2'),job('5',null,'file')])
  assert.deepEqual(edges.map(({from,to})=>[from,to]),[['1','2'],['1','3'],['2','4']])
  assert.equal(edges[0].lane,0)
  assert.notEqual(edges[1].lane,edges[2].lane)
})

test('missing or forward references do not produce misleading lines',()=>{
  assert.deepEqual(dependencyEdges([job('1','3'),job('2','missing'),job('3')]),[])
})

test('adjacent dependencies are straight; skip-row dependencies bend away from other nodes',()=>{
  const edges=dependencyEdges([job('1'),job('2','1'),job('3','1')])
  assert.equal(dependencyPath(edges[0],{x:54,y:100},{x:54,y:200}),'M 54 109 L 54 191')
  assert.match(dependencyPath(edges[1],{x:54,y:100},{x:54,y:300}),/^M 54 109 C /)
})

test('context menus stay inside viewport at bottom and right edges',()=>{
  assert.deepEqual(menuPosition({x:1190,y:790},{width:246,height:320},{width:1200,height:800}),{x:944,y:470})
  assert.deepEqual(menuPosition({x:-10,y:-10},{width:246,height:320},{width:1200,height:800}),{x:10,y:10})
})

test('move actions respect dependency order and running workflows',()=>{
  const jobs=[job('1'),job('2','1'),job('3')]
  assert.equal(canMoveJob(jobs,'2',-1),false)
  assert.equal(canMoveJob(jobs,'1',-1),false)
  assert.equal(canMoveJob(jobs,'3',-1),true)
  assert.equal(canMoveJob(jobs,'3',1),false)
  jobs[0].status='running'
  assert.equal(canMoveJob(jobs,'3',-1),false)
})
