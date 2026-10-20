"use strict";
const $=s=>document.querySelector(s), sessionsEl=$("#sessions"), messages=$("#messages"),
statusEl=$("#status"), requestStatus=$("#request-status"), form=$("#form"), input=$("#input"),
sendBtn=$("#send"), cancelBtn=$("#cancel"), inspectBtn=$("#inspect"), newBtn=$("#new");
const KEY="day20-client-id", clientId=localStorage.getItem(KEY)||`browser-${crypto.randomUUID()}`;
localStorage.setItem(KEY,clientId);
let ws=null,currentSession=null,activeRequest=null,answerEl=null,retry=0,timer=null,heartbeat=null;

function message(role,text,kind){const box=document.createElement("div");box.className=`msg ${kind}`;
box.textContent=`${role}：${text}`;messages.append(box);messages.scrollTop=messages.scrollHeight;return box}
function open(){return ws&&ws.readyState===WebSocket.OPEN}
function controls(){const ready=open()&&currentSession&&!activeRequest;input.disabled=!ready;sendBtn.disabled=!ready;
cancelBtn.disabled=!open()||!activeRequest;inspectBtn.disabled=!open();newBtn.disabled=!open();if(ready)input.focus()}
function send(type,id,payload={}){if(open())ws.send(JSON.stringify({version:"1.0",type,request_id:id,payload}))}
function renderSessions(items){sessionsEl.replaceChildren();for(const item of items){const b=document.createElement("button");
b.textContent=item.title;b.title=item.session_id;if(item.session_id===currentSession)b.classList.add("active");
b.onclick=()=>send("switch_session",`switch-${crypto.randomUUID()}`,{session_id:item.session_id});sessionsEl.append(b)}}
function renderTurns(turns){messages.replaceChildren();for(const t of turns){message("你",t.question,"user");message("助手",t.answer,"assistant")}}
function state(payload){currentSession=payload.current_session_id;renderSessions(payload.sessions||[]);renderTurns(payload.turns||[]);
requestStatus.textContent=currentSession?`当前会话：${currentSession}`:"请创建新对话。";controls()}
function finish(){activeRequest=null;answerEl=null;controls()}
function handle(m){const p=m.payload||{};
if(m.type==="connected"||m.type==="state")state(p);
else if(m.type==="queued")requestStatus.textContent="模型请求排队中……";
else if(m.type==="start"){requestStatus.textContent=`模型回答中，加载${p.history_turns}轮历史`;answerEl=message("助手","","assistant")}
else if(m.type==="token"&&answerEl){answerEl.textContent+=p.text;messages.scrollTop=messages.scrollHeight}
else if(m.type==="done"){requestStatus.textContent=`已保存到SQLite，turn_id=${p.turn.turn_id}`;finish();send("list_sessions",`list-${crypto.randomUUID()}`)}
else if(m.type==="duplicate"){message("助手",p.turn.answer,"assistant");requestStatus.textContent="重复request_id，直接返回数据库结果。";finish()}
else if(m.type==="cancelled"){requestStatus.textContent="已取消，部分回答未保存。";finish()}
else if(m.type==="status")message("状态",JSON.stringify(p,null,2),"system");
else if(m.type==="error"||m.type==="busy"||m.type==="cancel_not_found"){message("错误",p.message||JSON.stringify(p),"error");if(m.request_id===activeRequest)finish()}}
function connect(){statusEl.textContent="连接中";ws=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws?client_id=${encodeURIComponent(clientId)}`);
ws.onopen=()=>{retry=0;statusEl.textContent="已连接";clearInterval(heartbeat);heartbeat=setInterval(()=>send("ping",`ping-${crypto.randomUUID()}`),10000);controls()};
ws.onmessage=e=>{try{handle(JSON.parse(e.data))}catch(err){message("前端错误",String(err),"error")}};
ws.onclose=()=>{clearInterval(heartbeat);statusEl.textContent="已断开，等待重连";activeRequest=null;answerEl=null;controls();const wait=Math.min(2**retry++,8)*1000;clearTimeout(timer);timer=setTimeout(connect,wait)};
ws.onerror=()=>statusEl.textContent="连接错误"}
newBtn.onclick=()=>{const title=prompt("新对话标题：","新对话");if(title!==null)send("create_session",`create-${crypto.randomUUID()}`,{title})};
inspectBtn.onclick=()=>send("status",`status-${crypto.randomUUID()}`);
cancelBtn.onclick=()=>{if(activeRequest)send("cancel",activeRequest)};
form.onsubmit=e=>{e.preventDefault();const q=input.value.trim();if(!q||activeRequest)return;activeRequest=`req-${crypto.randomUUID()}`;message("你",q,"user");send("ask",activeRequest,{question:q});input.value="";controls()};
input.onkeydown=e=>{if(e.ctrlKey&&e.key==="Enter")form.requestSubmit()};connect();
