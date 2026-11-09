(()=>{
 const bar=document.querySelector('.progress i');
 const update=()=>{const d=document.documentElement;bar.style.width=(d.scrollTop/Math.max(1,d.scrollHeight-d.clientHeight)*100)+'%'}; addEventListener('scroll',update,{passive:true});update();
 const links=[...document.querySelectorAll('.toc a')],heads=links.map(a=>document.getElementById(a.hash.slice(1))).filter(Boolean); if('IntersectionObserver'in window){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id))}),{rootMargin:'-15% 0px -70%'});heads.forEach(h=>io.observe(h))}
 document.querySelectorAll('[data-motion]:not([data-scroll-motion])').forEach(fig=>{const btn=fig.querySelector('button'),img=fig.querySelector('.static-frame'),video=fig.querySelector('video'),anim=fig.dataset.animation,stat=fig.dataset.static,auto=fig.dataset.autoplay==='true';if(!btn||!img)return;let playing=false,fallbackActive=false;const reduced=matchMedia('(prefers-reduced-motion: reduce)');const set=(next)=>{playing=next;fallbackActive=false;if(video){img.hidden=playing;video.hidden=!playing;if(!playing){video.pause();video.currentTime=0;img.src=stat}}else if(anim){img.src=playing?anim:stat}btn.textContent=playing?(auto?'アニメーションを停止':'停止して静止画を表示'):(auto?'アニメーションを再開':'アニメーションを再生');btn.setAttribute('aria-pressed',String(playing))};const fallback=()=>{if(!auto||!anim||reduced.matches)return set(false);fallbackActive=true;if(video){video.pause();video.hidden=true}img.src=anim;img.hidden=false;playing=true;btn.textContent='アニメーションを停止';btn.setAttribute('aria-pressed','true')};const start=()=>{set(true);if(video)video.play().catch(fallback)};btn.addEventListener('click',()=>{if(!playing)start();else set(false)});if(video){video.addEventListener('ended',()=>{if(!auto)set(false)});video.addEventListener('error',fallback)}if(auto){const onReducedMotionChange=e=>{if(e.matches)set(false);else if(!playing&&!fallbackActive)start()};if(reduced.addEventListener)reduced.addEventListener('change',onReducedMotionChange);else reduced.addListener(onReducedMotionChange)}if(auto&&!reduced.matches)start();else set(false)});

 // A mathematical plate plays once, only after the reader settles on its art.
 // Observe the art rather than the caption: long captions must not delay playback.
 const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
 const plates=[...document.querySelectorAll('[data-scroll-motion]')].map(fig=>({
   fig, art:fig.querySelector('.figure-art'), video:fig.querySelector('video'),
   image:fig.querySelector('.static-frame'), button:fig.querySelector('.motion button'),
   note:fig.querySelector('.motion span'), started:false, completed:false,
   blocked:false, playing:false, resumePending:false, request:0
 })).filter(s=>s.art&&s.video&&s.image&&s.button);
 const dwellMs=600;
 let activePlate=null, pendingPlate=null, dwellTimer=null, scheduledFrame=null;
 const refreshPlate=s=>{
   s.button.textContent=s.playing?'停止して静止画を表示':s.completed?'もう一度再生':
     s.resumePending?'アニメーションを再開':'アニメーションを再生';
   s.button.setAttribute('aria-pressed',String(s.playing));
   if(s.note)s.note.textContent=reducedMotion.matches?
     '静止画を表示 · 必要なときだけ再生できます':'図が見えたら一度だけ自動再生 · いつでも停止できます';
 };
 const stopPlate=(s,{reset=false,block=false}={})=>{
   ++s.request; s.video.pause(); s.playing=false;
   if(block)s.blocked=true;
   s.resumePending=!reset&&s.started&&!s.completed;
   if(reset){s.video.currentTime=0;s.video.hidden=true;s.image.hidden=false;}
   if(activePlate===s)activePlate=null;
   refreshPlate(s);
 };
 const clearDwell=()=>{
   clearTimeout(dwellTimer);dwellTimer=null;pendingPlate=null;
 };
 const visibleFraction=s=>{
   const r=s.art.getBoundingClientRect();
   const available=Math.max(1,innerHeight-48);
   const visible=Math.max(0,Math.min(r.bottom,innerHeight-24)-Math.max(r.top,24));
   return r.width>0&&r.height>0&&r.right>0&&r.left<innerWidth?
     visible/Math.min(r.height,available):0;
 };
 const eligible=s=>!document.hidden&&!reducedMotion.matches&&!s.blocked&&
   !s.completed&&!s.playing&&(!s.started||s.resumePending)&&visibleFraction(s)>=.6;
 const startPlate=(s,automatic)=>{
   if(document.hidden||(automatic&&!eligible(s)))return;
   clearDwell();
   if(activePlate&&activePlate!==s)stopPlate(activePlate);
   s.video.muted=true;s.video.loop=false;
   s.started=true;s.playing=true;s.resumePending=false;
   s.video.hidden=false;s.image.hidden=true;activePlate=s;
   const request=++s.request;
   refreshPlate(s);
   // Autoplay can be blocked by browser policy. Keep a usable still and button;
   // do not substitute an uncontrollable, looping GIF or retry on every scroll.
   s.video.play().catch(()=>{
     if(s.request===request&&s.playing)stopPlate(s,{reset:true,block:true});
   });
 };
 const updatePlates=()=>{
   scheduledFrame=null;
   if(document.hidden){clearDwell();plates.forEach(s=>{if(s.playing)stopPlate(s);});return;}
   // Hysteresis avoids flicker near the start threshold while scrolling slowly.
   if(activePlate&&visibleFraction(activePlate)<.25)stopPlate(activePlate);
   if(activePlate||reducedMotion.matches){clearDwell();return;}
   const candidates=plates.filter(eligible).sort((a,b)=>{
     const center=s=>{const r=s.art.getBoundingClientRect();return Math.abs((r.top+r.bottom)/2-innerHeight/2);};
     return center(a)-center(b);
   });
   const next=candidates[0];
   if(!next){clearDwell();return;}
   if(pendingPlate===next)return;
   clearDwell();pendingPlate=next;
   dwellTimer=setTimeout(()=>{
     clearDwell();
     if(!activePlate&&eligible(next))startPlate(next,true);
   },dwellMs);
 };
 const schedulePlates=()=>{
   if(scheduledFrame===null)scheduledFrame=requestAnimationFrame(updatePlates);
 };
 plates.forEach(s=>{
   s.video.loop=false;s.video.muted=true;
   s.video.hidden=true;s.image.hidden=false;
   s.button.addEventListener('click',()=>{
     clearDwell();
     if(s.playing){stopPlate(s,{reset:true,block:true});return;}
     // A deliberate pause/replay permanently takes this plate out of auto mode.
     s.blocked=true;s.completed=false;
     if(visibleFraction(s)<.25)s.art.scrollIntoView({block:'center',behavior:'instant'});
     startPlate(s,false);
   });
   s.video.addEventListener('ended',()=>{
     s.completed=true;stopPlate(s,{reset:true});schedulePlates();
   });
   s.video.addEventListener('error',()=>stopPlate(s,{reset:true,block:true}));
   s.image.addEventListener('load',schedulePlates);
   refreshPlate(s);
 });
 const preferenceChanged=()=>{
   clearDwell();
   plates.forEach(s=>{
     if(reducedMotion.matches&&s.playing)stopPlate(s,{reset:true,block:true});
     refreshPlate(s);
   });
   schedulePlates();
 };
 if(reducedMotion.addEventListener)reducedMotion.addEventListener('change',preferenceChanged);
 else reducedMotion.addListener(preferenceChanged);
 addEventListener('scroll',schedulePlates,{passive:true});
 addEventListener('resize',schedulePlates,{passive:true});
 document.addEventListener('visibilitychange',()=>{
   // requestAnimationFrame may be suspended in a background tab.
   if(document.hidden)updatePlates();else schedulePlates();
 });
 addEventListener('pagehide',()=>{clearDwell();plates.forEach(s=>{if(s.playing)stopPlate(s);});});
 addEventListener('pageshow',schedulePlates);
 schedulePlates();
 const gate=document.querySelector('[data-gate]');if(gate){const r=gate.querySelector('input'),o=gate.querySelector('output'),a=gate.querySelector('[data-arrow]'),n=gate.querySelector('[data-negative]');const draw=()=>{const s=Number(r.value);o.textContent=s.toFixed(1);a.setAttribute('y2',75-s*26);a.dataset.sign=s<0?'negative':'positive';a.style.stroke=s<0?'#9892c7':'#4c83be';a.style.opacity=s===0?'0':'1';n.textContent=s<0?'負のゲート：指令の向きが反転':s>0?'正のゲート：指令をそのまま適用':'ゲートがゼロ：変化なし'};r.addEventListener('input',draw);draw()}
 const metric=document.querySelector('[data-metric]');if(metric){const x=metric.querySelector('[data-x]'),y=metric.querySelector('[data-v]'),o=metric.querySelector('[data-proj]'),v=metric.querySelector('[data-vector]'),p=metric.querySelector('[data-projection]'),pt=metric.querySelector('[data-point]'),pp=metric.querySelector('[data-projected-point]');const draw=()=>{const vx=Number(x.value),vy=Number(y.value),sx=260+vx*70,sy=130-vy*22,px=260+(vx+.25*vy)*70;o.textContent=`P(v)=(${(vx+.25*vy).toFixed(2)}, 0)`;v.setAttribute('d',`M260 130L${sx} ${sy}`);p.setAttribute('d',`M260 130L${px} 130`);pt.setAttribute('cx',sx);pt.setAttribute('cy',sy);pp.setAttribute('cx',px)};x.addEventListener('input',draw);y.addEventListener('input',draw);draw()}
})();
