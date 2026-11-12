(()=>{
'use strict';
const bar=document.querySelector('.progress');
const update=()=>{const d=document.documentElement;if(bar)bar.style.width=(d.scrollTop/Math.max(1,d.scrollHeight-d.clientHeight)*100)+'%'};
addEventListener('scroll',update,{passive:true});update();
const hero=document.querySelector('[data-hero]');
if(hero){
  const canvas=hero.querySelector('[data-hero-canvas]'),controls=hero.querySelector('[data-hero-controls]'),toggle=hero.querySelector('[data-hero-toggle]'),replay=hero.querySelector('[data-hero-replay]'),compare=hero.querySelector('[data-hero-compare]'),phase=hero.querySelector('[data-hero-phase]');
  let reduced=matchMedia('(prefers-reduced-motion: reduce)').matches, userPaused=false, inView=false, player=null;
  const setButton=()=>{if(!toggle||!player)return; const ended=player.ended; toggle.textContent=ended?'Replay animation':player.paused?'Play animation':'Pause animation'; toggle.setAttribute('aria-pressed',String(!player.paused&&!ended)); if(replay)replay.disabled=!ended&&player.currentTime<.1};
  const setPhase=()=>{if(!phase||!player)return;const t=player.currentTime;phase.textContent=t<4?'Act I: both ledgers agree in advance that 17 means a red circle.':t<14?'Act I: the same Morse numeral 17 reaches the receiver. The shape ledger was agreed before transmission.':t<18.1?'17 arrived intact; both ledgers say red circle.':t<24.5?'Act II: the books remain stationary; the receiver applies a local rule that reads the unchanged 17 as a red five-point star.':t<35?'The sender transmits 17 again. The Morse convention stays shared; the receiver’s shape rule differs.':t<40?'Sender: red circle. Receiver: red five-point star. The transmitted numeral is unchanged.':'Successful signal delivery does not establish shared interpretation. Learned semantic correspondence needs separate evidence.'};
  const listen=()=>{['timeupdate','seeked','play','pause','ended'].forEach(event=>player.on(event,()=>{setPhase();setButton();hero.classList.toggle('is-playing',!player.paused&&!player.ended)}))};
  const reveal=()=>{hero.classList.add('is-ready');if(controls)controls.hidden=false;setPhase();setButton()};
  const fail=()=>{hero.classList.add('is-failed');if(controls)controls.hidden=false};
  const play=()=>{if(!player||reduced||userPaused||!inView)return;try{const attempt=player.play();if(attempt&&typeof attempt.catch==='function')attempt.catch(fail)}catch(error){fail(error)}};
  const mount=window.TelegramAnimation&&window.TelegramAnimation.mount;
  if(!canvas||!mount){fail()}else mount(canvas,{assetBase:'assets/telegram/'}).then(instance=>{player=window.telegramPlayer=instance;listen();reveal();if(reduced)player.pause();else play()}).catch(fail);
  if(toggle)toggle.addEventListener('click',()=>{if(!player)return;if(player.ended){player.seek(0);userPaused=false;player.play()}else if(player.paused){userPaused=false;player.play()}else{userPaused=true;player.pause()}setButton()});
  if(replay)replay.addEventListener('click',()=>{if(player){userPaused=false;player.seek(0);player.play();setButton()}});
  if(compare)compare.addEventListener('click',()=>{if(player){userPaused=false;player.seek(18.1);player.play();setPhase()}});
  const observer='IntersectionObserver' in window?new IntersectionObserver(entries=>{inView=entries[0].isIntersecting&&entries[0].intersectionRatio>=.2;if(inView)play();else if(player&&!player.paused)player.pause()},{threshold:[0,.2]}):null;
  if(observer)observer.observe(hero); else {inView=true;play()}
  addEventListener('visibilitychange',()=>{if(document.hidden&&player&&!player.paused)player.pause();else if(!document.hidden&&inView&&!userPaused)play()});
  const motionQuery=matchMedia('(prefers-reduced-motion: reduce)');
  const motionChanged=e=>{reduced=e.matches;if(reduced&&player)player.pause();else if(inView&&!userPaused)play()};
  if(motionQuery.addEventListener)motionQuery.addEventListener('change',motionChanged);else motionQuery.addListener(motionChanged);
}
const panel=document.querySelector('[data-cancellation]');if(!panel)return;
const U={U00:x=>x[0],U10:x=>x[1],U01:x=>1-x[0],U11:x=>1-x[1]};
const formulas={U00:'xA',U10:'xB',U01:'1 − xA',U11:'1 − xB'};
const outcomes=[[0,0],[0,1],[1,0],[1,1]],original=[[1,0],[0,1]],replacement=[[1,1],[0,0]];
const pair=x=>`(${x[0]}, ${x[1]})`,equal=(a,b)=>a.every((v,i)=>v===b[i]);
const argmax=(menu,u)=>menu.reduce((best,item)=>u(item)>u(best)?item:best,menu[0]);
const describe=x=>x[0]?(x[1]?'Both survive':'A survives; B is lost'):(x[1]?'B survives; A is lost':'Both are lost');
function transform(x,flip){return flip==='A'?[1-x[0],x[1]]:flip==='B'?[x[0],1-x[1]]:x}
let chosen='U00',diagnostic='none';
function draw(){
  panel.querySelectorAll('[data-util]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.util===chosen)));
  panel.querySelectorAll('[data-diagnostic]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.diagnostic===diagnostic)));
  const u=U[chosen];
  panel.querySelector('[data-rows]').innerHTML=outcomes.map(x=>`<tr><td>${pair(x)}</td><td>${u(x)}</td><td>${x[0]}</td></tr>`).join('');
  panel.querySelector('[data-menu-results]').innerHTML=[['Original test',original],['Replacement choice set',replacement]].map(([label,menu])=>{
    const choice=argmax(menu,u);
    return `<div class="menu-card"><strong>${label}</strong><small>Options: ${menu.map(pair).join(' or ')}</small><span class="choice">${describe(choice)}</span><span>Human score: <b>${choice[0]}</b></span></div>`;
  }).join('');
  const repairs=[['None','U11'],['Target only','U01'],['Preference only','U10'],['Both','U00']];
  panel.querySelector('[data-repairs]').innerHTML='<table class="finite-table"><thead><tr><th>Repair</th><th>Resulting rule</th><th>Choice</th><th>Human score</th></tr></thead><tbody>'+repairs.map(([label,name])=>{
    const choice=argmax(original,U[name]);
    return `<tr><td>${label}</td><td>${formulas[name]}</td><td>${pair(choice)}</td><td class="${choice[0]?'pass':'fail'}">${choice[0]} · ${choice[0]?'Pass':'Fail'}</td></tr>`;
  }).join('')+'</tbody></table>';
  panel.querySelector('[data-summary]').textContent=`Selected rule: ${chosen}(x) = ${formulas[chosen]}. The repair table always starts from the agent with both errors (U11).`;
  const baseline=argmax(original,u),choice=argmax(original,x=>u(transform(x,diagnostic)));
  panel.querySelector('[data-diagnostic-output]').textContent=`${diagnostic==='none'?'Original signals restored':`Only the ${diagnostic} signal is flipped for both candidate outcomes`}. ${chosen} chooses ${pair(choice)}: ${describe(choice)}. ${equal(choice,baseline)?'Choice unchanged.':'Choice reversed.'} Physical options remain (1, 0) and (0, 1).`;
}
panel.querySelectorAll('[data-util]').forEach(b=>b.addEventListener('click',()=>{chosen=b.dataset.util;diagnostic='none';draw()}));
panel.querySelectorAll('[data-diagnostic]').forEach(b=>b.addEventListener('click',()=>{diagnostic=b.dataset.diagnostic;draw()}));draw();
})();
