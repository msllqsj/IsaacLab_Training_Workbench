import { test } from 'node:test'
import assert from 'node:assert/strict'
import { latestRequest } from '../src/latestRequest.ts'

function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no});return {promise,resolve,reject}}

test('a slow pre-deletion snapshot cannot restore a deleted history row',async()=>{
  const visible=[],gate=latestRequest(snapshot=>visible.push(snapshot)),old=deferred()
  const stale=gate.run(()=>old.promise)
  gate.invalidate()
  await gate.run(()=>Promise.resolve({runs:[]}),true)
  old.resolve({runs:[{id:'deleted'}]});await stale
  assert.deepEqual(visible,[{runs:[]}])
})

test('clearing logs invalidates an in-flight response containing old text',async()=>{
  const text=[],gate=latestRequest(page=>text.push(page.text)),old=deferred()
  const stale=gate.run(()=>old.promise)
  gate.invalidate()
  old.resolve({text:'old logs'});await stale
  await gate.run(()=>Promise.resolve({text:'new logs'}))
  assert.deepEqual(text,['new logs'])
})

test('ordinary polling coalesces overlapping requests without starving the first response',async()=>{
  const accepted=[],gate=latestRequest(value=>accepted.push(value)),load=deferred()
  const pending=gate.run(()=>load.promise)
  let duplicate=false
  await gate.run(()=>{duplicate=true;return Promise.resolve('wrong')})
  load.resolve('current');await pending
  assert.equal(duplicate,false);assert.deepEqual(accepted,['current'])
})

test('errors from invalidated requests are ignored; current errors are reported',async()=>{
  const gate=latestRequest(()=>{}),load=deferred()
  const pending=gate.run(()=>load.promise)
  gate.invalidate();load.reject(new Error('stale'));await pending
  await assert.rejects(gate.run(()=>Promise.reject(new Error('current'))),/current/)
})

test('invalidation lets fresh polling continue without waiting for a hung old request',async()=>{
  const values=[],gate=latestRequest(value=>values.push(value)),old=deferred()
  const stale=gate.run(()=>old.promise)
  gate.invalidate()
  await gate.run(()=>Promise.resolve('fresh'))
  assert.deepEqual(values,['fresh'])
  old.resolve('old');await stale
  assert.deepEqual(values,['fresh'])
})
