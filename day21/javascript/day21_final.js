"use strict";const $=s=>document.querySelector(s),connection=$("#connection"),messages=$("#messages"),sources=$("#sources"),
question=$("#question"),askForm=$("#ask-form"),askBtn=$("#ask"),cancelBtn=$("#cancel"),requestStatus=$("#request-status"),
uploadForm=$("#upload-form"),file=$("#file"),uploadBtn=$("#upload"),uploadStatus=$("#upload-status");
let ws=null,active=null,answer=null,retry=0;
function msg(role,text,kind){const e=document.createElement("div");e.className=`msg ${kind}`;e.textContent=`${role}：${text}`;messages.append(e);messages.scrollTop=messages.scrollHeight;return e}
function controls(){const online=ws&&ws.readyState===WebSocket.OPEN;question.disabled=!online||!!active;askBtn.disabled=!online||!!active;cancelBtn.disabled=!online||!active}
function send(type,id,payload={}){ws.send(JSON.stringify({version:"1.0",type,request_id:id,payload}))}
function finish(){active=null;answer=null;controls();question.focus()}
function renderSources(items){sources.replaceChildren();for(const item of items){const box=document.createElement("article");box.className="source";const h=document.createElement("strong");h.textContent=`资料${item.rank}｜${item.location}｜距离${item.distance}`;const p=document.createElement("p");p.textContent=item.content;box.append(h,p);sources.append(box)}}
function handle(m){const p=m.payload||{};if(m.type==="retrieving")requestStatus.textContent="正在检索知识库……";
else if(m.type==="retrieval"){renderSources(p.results||[]);requestStatus.textContent=`最佳距离${p.best_distance}，阈值${p.threshold}`}
else if(m.type==="rejected"){msg("系统",p.message,"error");requestStatus.textContent="检索守门拒绝回答。";finish()}
else if(m.type==="queued")requestStatus.textContent="资料通过守门，模型排队中……";
else if(m.type==="start"){requestStatus.textContent="模型正在根据资料回答……";answer=msg("助手","","assistant")}
else if(m.type==="token"&&answer){answer.textContent+=p.text;messages.scrollTop=messages.scrollHeight}
else if(m.type==="done"){requestStatus.textContent="回答完成，可核对下方来源。";renderSources(p.sources||[]);finish()}
else if(m.type==="cancelled"){requestStatus.textContent="回答已取消。";finish()}
else if(m.type==="error"||m.type==="busy"||m.type==="cancel_not_found"){msg("错误",p.message||JSON.stringify(p),"error");if(m.request_id===active)finish()}}
function connect(){ws=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws`);ws.onopen=()=>{retry=0;connection.textContent="已连接";controls()};ws.onmessage=e=>handle(JSON.parse(e.data));ws.onclose=()=>{connection.textContent="已断开，自动重连";active=null;answer=null;controls();setTimeout(connect,Math.min(2**retry++,8)*1000)};ws.onerror=()=>connection.textContent="连接错误"}
askForm.onsubmit=e=>{e.preventDefault();const q=question.value.trim();if(!q||active)return;active=`req-${crypto.randomUUID()}`;msg("你",q,"user");sources.replaceChildren();send("ask",active,{question:q});question.value="";controls()};
cancelBtn.onclick=()=>{if(active)send("cancel",active)};
uploadForm.onsubmit=async e=>{e.preventDefault();const selected=file.files[0];if(!selected)return;const body=new FormData();body.append("document",selected);uploadBtn.disabled=true;uploadStatus.textContent="上传、校验并同步索引中……";try{const response=await fetch("/upload",{method:"POST",body});const data=await response.json();if(!response.ok)throw new Error(data.message);uploadStatus.textContent=`${data.status}：${data.original_name}，索引已同步。`;uploadForm.reset()}catch(error){uploadStatus.textContent=`上传失败：${error}`}finally{uploadBtn.disabled=false}};connect();
