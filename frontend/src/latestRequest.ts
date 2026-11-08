export function latestRequest<T>(accept:(value:T)=>void){
  let version=0,pending=0
  return {
    invalidate(){version++;pending=0},
    async run(load:()=>Promise<T>,force=false){
      if(pending&&!force)return
      const current=++version
      pending=current
      try{
        const value=await load()
        if(current===version)accept(value)
      }catch(error){if(current===version)throw error}
      finally{if(pending===current)pending=0}
    },
  }
}
