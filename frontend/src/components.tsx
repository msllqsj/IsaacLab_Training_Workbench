import { useEffect, useState, type ReactNode, type ButtonHTMLAttributes } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { X, Folder, FileBox, ChevronUp, Loader2 } from 'lucide-react'
import { api, type Project } from './types'

export function Button({variant='default',className='',...props}:ButtonHTMLAttributes<HTMLButtonElement>&{variant?:'default'|'primary'|'danger'|'ghost'}){return <button className={`button ${variant} ${className}`} {...props}/>}
export function Modal({title,description,children,onClose,wide=false}:{title:string;description?:string;children:ReactNode;onClose:()=>void;wide?:boolean}){
 return <Dialog.Root open onOpenChange={open=>!open&&onClose()}><Dialog.Portal><Dialog.Overlay className="modal-overlay"/><Dialog.Content className={`modal ${wide?'wide':''}`} aria-describedby={description?'modal-desc':undefined}><div className="modal-heading"><div><Dialog.Title>{title}</Dialog.Title>{description&&<Dialog.Description id="modal-desc">{description}</Dialog.Description>}</div><Dialog.Close asChild><Button variant="ghost" aria-label="关闭"><X size={20}/></Button></Dialog.Close></div>{children}</Dialog.Content></Dialog.Portal></Dialog.Root>
}
export function Logo(){return <svg className="brand-logo" viewBox="0 0 64 64" role="img" aria-label="立方体工作台图标"><defs><linearGradient id="logo-a" x2="1" y2="1"><stop stopColor="var(--logo-hi)"/><stop offset="1" stopColor="var(--logo-main)"/></linearGradient><linearGradient id="logo-b" x2="0" y2="1"><stop stopColor="var(--logo-main)"/><stop offset="1" stopColor="var(--logo-deep)"/></linearGradient></defs><path d="M32 3 57 17 57 47 32 61 7 47 7 17Z" fill="url(#logo-b)"/><path d="M32 3 57 17 45 24 32 17 19 24 7 17Z" fill="url(#logo-a)"/><path d="M7 17 19 24 19 41 32 49 32 61 7 47Z" fill="var(--logo-main)"/><path d="M57 17 45 24 45 41 32 49 32 61 57 47Z" fill="var(--logo-deep)"/><path d="M32 20 44 27 32 34 20 27Z" fill="var(--logo-inner-top)"/><path d="M20 29 31 35 31 47 20 41Z" fill="var(--logo-inner-left)"/><path d="M33 35 44 29 44 41 33 47Z" fill="var(--logo-inner-right)"/></svg>}
export function Empty({icon,children,action}:{icon?:ReactNode;children:ReactNode;action?:ReactNode}){return <div className="empty">{icon}<div>{children}</div>{action}</div>}
type Entry={name:string;path:string;directory:boolean;size:number|null}
export function FilePicker({project,onSelect,onClose}:{project:Project;onSelect:(p:string)=>void;onClose:()=>void}){
 const [listing,setListing]=useState<{path:string;parent:string|null;roots:string[];entries:Entry[]}|null>(null)
 const [error,setError]=useState('')
 const load=async(path='')=>{try{setError('');setListing(await api(`/files?project_id=${encodeURIComponent(project.id)}&path=${encodeURIComponent(path)}`))}catch(e){setError(String(e))}}
 useEffect(()=>{void load()},[project.id])
 return <Modal title="选择 checkpoint" description="选择训练项目或工作台 outputs 中的 .pt 文件，也可在实验表单中直接填写其他完整路径。" onClose={onClose} wide><div className="file-picker"><div className="root-buttons">{listing?.roots.map((root,i)=><Button key={root} onClick={()=>void load(root)}><Folder size={15}/>{i===0?'训练项目':'工作台产物'}</Button>)}</div><div className="file-path"><Button variant="ghost" disabled={!listing?.parent} onClick={()=>listing?.parent&&void load(listing.parent)} aria-label="上一级"><ChevronUp size={18}/></Button><code>{listing?.path??'正在读取…'}</code></div>{error&&<div className="error-box">{error}</div>}<div className="file-list">{listing?.entries.map(f=><button key={f.path} className="file-entry" onClick={()=>f.directory?void load(f.path):(onSelect(f.path),onClose())}>{f.directory?<Folder size={18}/>:<FileBox size={18}/>}<span>{f.name}</span><small>{f.directory?'文件夹':`${((f.size??0)/1024/1024).toFixed(1)} MB`}</small></button>)}{listing?.entries.length===0&&<Empty>此目录暂无 checkpoint</Empty>}</div></div></Modal>
}
export function Busy({children}:{children:ReactNode}){return <span className="busy"><Loader2 className="spin" size={16}/>{children}</span>}
