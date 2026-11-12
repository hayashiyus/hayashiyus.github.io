/* The Same Telegram — action-authored Canvas2D film.
 * Each action evaluates an absolute time. No Figma timeline or video is used.
 * Watercolor sources are the approved painting and existing clean underpaintings.
 */
(function(global){
'use strict';
const WIDTH=1536,HEIGHT=1024,DURATION=48;
const C={
  "hand": "M99 414 C113 413 139 415 153 425 C171 440 181 455 194 473 L190 488 L175 490 L165 499 L152 496 L140 494 L126 489 L117 480 L109 458 L98 440 Z",
  "lever": "M174 474 L258 497 L261 509 L249 511 L173 485 Z",
  "paper": "M1092 438 C1110 452 1120 456 1145 453 C1193 450 1248 479 1303 501 L1345 518 L1363 534 C1354 542 1346 549 1330 551 C1311 551 1282 540 1245 528 L1121 483 C1104 473 1097 453 1092 438 Z",
  "handR": "M1332 515 C1348 514 1355 514 1372 523 L1400 542 L1422 563 L1426 579 L1404 590 L1383 579 L1366 558 L1347 554 L1322 550 L1315 538 L1324 530 Z",
  "receiverHead": "M1193 360 C1189 341 1193 337 1207 333 L1195 328 C1188 312 1194 303 1206 300 L1194 296 C1191 281 1196 275 1209 272 L1199 270 C1195 255 1202 248 1214 246 L1211 239 C1209 230 1219 224 1229 224 C1234 194 1247 178 1268 179 C1288 178 1303 191 1304 210 C1311 188 1323 176 1345 180 C1371 181 1385 201 1392 221 C1409 211 1425 212 1432 229 L1428 240 C1438 238 1441 244 1447 261 L1441 276 C1451 280 1455 289 1458 306 L1452 319 C1460 326 1462 345 1457 361 L1443 364 L1437 351 C1410 348 1395 350 1386 352 L1378 382 L1365 414 L1354 436 L1326 411 L1306 422 C1282 415 1264 394 1262 377 L1253 350 C1235 347 1220 349 1209 351 L1207 363 Z",
  "receiverClean": "M1178 170 H1465 V369 C1420 369 1390 379 1370 403 L1351 435 H1295 C1267 419 1239 390 1215 364 H1178 Z",
  "collar": "M1380 364 L1392 359 L1384 400 L1369 465 L1358 470 L1336 441 L1324 412 L1332 400 L1345 425 L1354 439 L1371 398 Z",
  "senderFace": "M320 73 C348 56 385 63 404 92 C411 121 407 157 385 182 L353 209 L316 210 C296 181 298 125 320 73 Z",
  "wire": "M271 508 C350 515 410 522 488 530 C510 535 534 547 552 559 L558 562 L564 567 L570 572 L576 577 L582 581 L588 584 L594 588 L600 593 L606 596 L612 599 L618 602 L624 605 L630 608 L636 610 L642 612 L648 615 L654 616 L660 618 L666 620 L672 621 L678 622 L684 623 L690 624 L696 625 L702 626 L708 626 L714 627 L720 627 L726 627 L732 626 L738 626 L744 626 L750 625 L756 624 L762 623 L768 622 L774 620 L780 619 L786 617 L792 615 L798 613 L804 611 L810 609 L816 607 L822 604 L828 601 L834 598 L840 595 L846 592 L852 588 L858 584 L864 580 L870 576 L876 571 L882 566 L888 561 L894 556 L900 552 L906 548 L912 544 C931 528 945 515 960 511"
};
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const smooth=v=>{v=clamp(v);return v*v*(3-2*v)};
const ramp=(t,a,b)=>smooth((t-a)/(b-a));
const fade=(t,a,b,c,d)=>ramp(t,a,b)*(1-ramp(t,c,d));
const mix=(a,b,q)=>a+(b-a)*q;
const EVENTS=[['.',1.2,1.44],['-',1.68,2.4],['-',2.64,3.36],['-',3.6,4.32],['-',4.56,5.28],['-',6,6.72],['-',6.96,7.68],['.',7.92,8.16],['.',8.4,8.64],['.',8.88,9.12]];
const EVENTS_ALL=[4,25].flatMap((off,pass)=>EVENTS.map(([mark,s,e],i)=>({mark,start:s+off,end:e+off,arrival:s+off+.72,arrivalEnd:e+off+.72,pass,index:i})));
function canvas(){const c=document.createElement('canvas');c.width=WIDTH;c.height=HEIGHT;return c}
// Apply the restrained paper wash to raster watercolor sources before masking.
// source-atop keeps any source alpha intact; vector strokes and captions are
// drawn later on the live context and therefore retain their original contrast.
function gradeWatercolor(image){
 const c=canvas(),x=c.getContext('2d');
 x.drawImage(image,0,0,WIDTH,HEIGHT);
 x.globalCompositeOperation='source-atop';x.globalAlpha=.055;x.fillStyle='#fbf8f2';x.fillRect(0,0,WIDTH,HEIGHT);
 x.globalAlpha=1;x.globalCompositeOperation='source-over';return c;
}
function masked(image,path,blur=0,stroke=0){
 const c=canvas(),x=c.getContext('2d');x.fillStyle='white';x.strokeStyle='white';x.lineJoin='round';
 if(blur)x.filter=`blur(${blur}px)`;
 const p=new Path2D(path);x.fill(p);if(stroke){x.lineWidth=stroke;x.stroke(p)}
 x.filter='none';x.globalCompositeOperation='source-in';x.drawImage(image,0,0,WIDTH,HEIGHT);return c;
}
function ellipseSprite(image,cx,cy,rx,ry){return masked(image,`M${cx-rx} ${cy} a${rx} ${ry} 0 1 0 ${2*rx} 0 a${rx} ${ry} 0 1 0 ${-2*rx} 0`,.35)}
function sample(path){const svg=document.createElementNS('http://www.w3.org/2000/svg','path');svg.setAttribute('d',path);const length=svg.getTotalLength();return Array.from({length:401},(_,i)=>{const p=svg.getPointAtLength(length*i/400);return[p.x,p.y]})}
function drawPart(ctx,sprite,dx=0,dy=0,angle=0,pivot=[0,0],scale=1){ctx.save();ctx.translate(pivot[0]+dx,pivot[1]+dy);ctx.rotate(angle);ctx.scale(scale,scale);ctx.translate(-pivot[0],-pivot[1]);ctx.drawImage(sprite,0,0);ctx.restore()}
function pathLine(ctx,points,a=0,b=1){const lo=Math.floor(clamp(a)*400),hi=Math.ceil(clamp(b)*400);if(hi<=lo)return;ctx.beginPath();for(let i=lo;i<=hi;i++){const p=points[i];if(i===lo)ctx.moveTo(...p);else ctx.lineTo(...p)}ctx.stroke()}
function text(ctx,s,x,y,size=28,opacity=1){if(opacity<=0)return;ctx.save();ctx.globalAlpha=opacity;ctx.fillStyle='#2c3e46';ctx.font=`${size}px Georgia, 'Times New Roman', serif`;ctx.textAlign='center';ctx.textBaseline='top';s.split('\n').forEach((line,i)=>ctx.fillText(line,x,y+i*size*1.32));ctx.restore()}
function iris(ctx,path,cx,cy,tx,ty,rx=3,ry=3.8){ctx.save();ctx.clip(new Path2D(path));ctx.fillStyle='#d4cfc1';ctx.fillRect(cx-20,cy-20,40,40);const dx=tx-cx,dy=ty-cy,n=Math.hypot(dx,dy,100),px=cx+5*dx/n,py=cy+3.1*dy/n;ctx.fillStyle='#53483c';ctx.beginPath();ctx.ellipse(px,py,rx+.6,ry+.6,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#282822';ctx.beginPath();ctx.ellipse(px,py,rx,ry,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#131b1c';ctx.beginPath();ctx.ellipse(px,py,rx*.5,ry*.65,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#dcd8c6';ctx.globalAlpha=.5;ctx.beginPath();ctx.arc(px-.7,py-1,.55,0,Math.PI*2);ctx.fill();ctx.restore()}
function interpolate(t,keys){let i=0;while(i<keys.length-2&&t>keys[i+1][0])i++;const a=keys[i],b=keys[i+1],q=ramp(t,a[0],b[0]);return [mix(a[1],b[1],q),mix(a[2],b[2],q)]}
class Film{
 constructor(canvas,images){this.canvas=canvas;this.ctx=canvas.getContext('2d',{alpha:false});this.images=images;this.listeners={};this.time=0;this.running=false;this.raf=0;this.build();this.render(0)}
 get duration(){return DURATION}get currentTime(){return this.time}get paused(){return !this.running}get ended(){return this.time>=DURATION}
 on(name,cb){(this.listeners[name]??=[]).push(cb);return()=>this.listeners[name]=this.listeners[name].filter(x=>x!==cb)}
 emit(name){(this.listeners[name]||[]).forEach(f=>f())}
 play(){if(this.running)return;if(this.ended)this.seek(0);this.running=true;this.origin=performance.now()-this.time*1000;this.emit('play');const tick=now=>{if(!this.running)return;this.time=Math.min(DURATION,(now-this.origin)/1000);this.render(this.time);this.emit('timeupdate');if(this.ended){this.running=false;this.emit('ended');this.emit('pause')}else this.raf=requestAnimationFrame(tick)};this.raf=requestAnimationFrame(tick)}
 pause(){if(!this.running)return;this.running=false;cancelAnimationFrame(this.raf);this.emit('pause')}
 seek(t){this.time=clamp(t,0,DURATION);this.origin=performance.now()-this.time*1000;this.render(this.time);this.emit('seeked');this.emit('timeupdate')}
 build(){
  const I=this.images,W={painting:gradeWatercolor(I.painting),cleanMotion:gradeWatercolor(I.cleanMotion),eyes:gradeWatercolor(I.eyes),cleanReceiver:gradeWatercolor(I.cleanReceiver),cleanSender:gradeWatercolor(I.cleanSender)};this.wirePoints=sample(C.wire);this.s={};
  // Exactly two original receiver hands. No synthetic arm/page-turn asset is loaded.
  const stage=canvas(),g=stage.getContext('2d');g.drawImage(W.painting,0,0);
  g.drawImage(masked(W.cleanMotion,'M89 404 H196 L272 493 V521 H160 L97 491 Z',1),0,0);
  g.drawImage(masked(W.cleanMotion,'M1045 425 H1450 V610 H1045 Z',1),0,0);
  g.drawImage(masked(W.cleanReceiver,C.receiverClean,5,18),0,0);
  g.drawImage(masked(W.cleanSender,C.senderFace,5),0,0);
  this.stage=stage;
  this.s.hand=masked(W.painting,C.hand,.8);this.s.lever=masked(W.painting,C.lever,.5);
  this.s.paper=masked(W.painting,C.paper,.4);this.s.handR=masked(W.painting,C.handR,.6);
  this.s.reel=masked(W.painting,'M1003 384 H1126 V435 L1105 440 L1115 460 L1130 475 L1118 491 H1003 Z',.3);
  this.s.head=masked(W.painting,C.receiverHead,.7,10);this.s.collar=masked(W.painting,C.collar,1);
  this.s.face=masked(W.eyes,C.senderFace,3);
  this.s.eyeClean=masked(W.eyes,'M1256 329 C1270 324 1289 328 1290 339 C1300 322 1315 316 1324 320 L1325 348 L1293 357 L1260 354 Z',1);
  // Existing watercolor pigments provide the grain of the animated dictionary marks.
  const spot=canvas(),p=spot.getContext('2d');p.drawImage(ellipseSprite(W.painting,218,713,17.7,17.1),0,0);
  this.s.red=canvas();const r=this.s.red.getContext('2d');r.drawImage(spot,0,0);r.globalCompositeOperation='color';r.fillStyle='#c8514f';r.fillRect(197,693,43,42);r.globalCompositeOperation='destination-in';r.drawImage(spot,0,0);
  this.s.triangle=masked(W.painting,'M1169 714 L1180 742 L1150 735 Z',.3);
  this.s.square=masked(W.painting,'M1140 761 L1165 768 L1155 793 L1127 785 Z',.3);
  this.bookBase=canvas();const b=this.bookBase.getContext('2d');
  // Blank paper from the same painted page: erase only the three old color marks.
  [[1167,729],[1148,778],[1116,834]].forEach(([x,y])=>{
   b.save();b.beginPath();b.ellipse(x,y,26,24,-.1,0,Math.PI*2);b.clip();b.drawImage(W.painting,1200,767,56,52,x-28,y-26,56,52);b.restore();
  });
 }
 drawStage(){this.ctx.drawImage(this.stage,0,0)}
 // Action 1: the sender's fingertips and the brass lever share the same Morse gate.
 senderKey(t){let gate=0;for(const e of EVENTS_ALL)gate=Math.max(gate,fade(t,e.start-.07,e.start,e.end,e.end+.10));const y=-4.2*(1-gate);drawPart(this.ctx,this.s.lever,0,y*.46);drawPart(this.ctx,this.s.hand,0,y)}
 // Action 2: a finite bright segment travels on the actual cable, delayed by .72 s.
 cableSignal(t){const c=this.ctx;for(const e of EVENTS_ALL){if(t<e.start||t>e.arrivalEnd+.1)continue;const a=(t-e.end)/.72,b=(t-e.start)/.72;c.save();c.lineCap='round';c.lineJoin='round';c.globalAlpha=.78*(1-ramp(t,e.arrivalEnd,e.arrivalEnd+.1));c.strokeStyle='#e2b566';c.shadowColor='#edb557';c.shadowBlur=12;c.lineWidth=4;pathLine(c,this.wirePoints,a,b);c.shadowBlur=0;c.strokeStyle='#fff0bd';c.lineWidth=1.9;pathLine(c,this.wirePoints,a,b);c.restore()}}
 feed(t){return 18*(clamp(t-5.92,0,7.92)+clamp(t-26.92,0,7.92))}
 // Action 3: paper moves out of the register; ink is deposited at its fixed nib.
 receiverPaper(t){const c=this.ctx,feed=this.feed(t),q=feed/285.12,extension=.23*q;
  c.save();c.translate(1118,476);c.transform(1+extension*.943*.943,extension*.943*.333,extension*.943*.333,1+extension*.333*.333,0,0);c.translate(-1118,-476);c.drawImage(this.s.paper,0,0);c.restore();
  const unit=[.943,.333],nib=[1125,468];
  c.save();c.strokeStyle='#413e32';c.lineCap='round';c.lineWidth=2.1;
  for(const e of EVENTS_ALL){if(t<e.arrival)continue;const start=this.feed(e.arrival),end=this.feed(Math.min(t,e.arrivalEnd));const a=feed-end,b=feed-start;
   c.beginPath();c.moveTo(nib[0]+unit[0]*a,nib[1]+unit[1]*a);c.lineTo(nib[0]+unit[0]*b,nib[1]+unit[1]*b);c.stroke();}
  c.restore();
  drawPart(c,this.s.reel);drawPart(c,this.s.handR,51*q,18*q);
  // The left telegram records the exact international Morse numeral 17.
  c.save();c.strokeStyle='#443f33';c.lineWidth=2.1;c.lineCap='round';let x=266,y=588;
  for(let i=0;i<EVENTS.length;i++){const w=EVENTS[i][0]==='.'?2:12;c.beginPath();c.moveTo(x,y);c.lineTo(x+w,y-w*.12);c.stroke();x+=w+6+(i===4?10:0);y=588-(x-266)*.12}c.restore();
 }
 // Action 4: eyes lead the receiver's head; head follows tape, then the codebook.
 receiverTarget(t){
  // Follow the most recently printed mark, not the old free end of the strip.
  for(const [start,end] of [[5.72,14.1],[26.72,35.1]])if(t>=start&&t<=end){
   const latest=EVENTS_ALL.filter(e=>e.arrival<=t).at(-1);
   if(!latest)return [1125,468];
   const inkCenter=(this.feed(latest.arrival)+this.feed(Math.min(t,latest.arrivalEnd)))/2;
   const distance=this.feed(t)-inkCenter;return [1125+.943*distance,468+.333*distance];
  }
  return interpolate(t,[[0,1167,729],[3,1167,729],[5.72,1125,468],[14.1,1129,470],[15,1245,512],[16.3,1167,729],[24.5,1167,729],[26.72,1125,468],[35.1,1129,470],[35.8,1295,530],[37,1167,729],[48,1167,729]]);
 }
 receiverGaze(t){
  const target=this.receiverTarget(t),late=this.receiverTarget(Math.max(0,t-.23)),q=clamp((late[1]-500)/229),a=(-2.3+5*q)*Math.PI/180,p=[1330,420];
  const c=this.ctx;c.save();c.translate(p[0]+1.5*q,p[1]+1.2*q);c.rotate(-a);c.translate(-p[0],-p[1]);c.drawImage(this.s.head,0,0);c.drawImage(this.s.eyeClean,0,0);
  const dx=target[0]-p[0],dy=target[1]-p[1],tx=p[0]+Math.cos(a)*dx-Math.sin(a)*dy,ty=p[1]+Math.sin(a)*dx+Math.cos(a)*dy;
  iris(c,'M1265 339 C1271 339 1277 340 1282 342 C1276 347 1270 348 1265 339 Z',1274,342,tx,ty,2.9,3.5);
  iris(c,'M1298 339 C1301 334 1310 331 1317 327 C1318 337 1309 343 1298 342 Z',1306,336,tx,ty,3,3.6);
  c.restore();c.drawImage(this.s.collar,0,0);
 }
 // Action 5: one brief confirmation glance after each transmission; otherwise watch key.
 senderGlance(t){let q=0,eye=0;for(const off of[4,25]){q=Math.max(q,fade(t,off+9.31,off+9.76,off+10.25,off+10.94));eye=Math.max(eye,fade(t,off+9.22,off+9.42,off+10.19,off+10.5))}
  const c=this.ctx;c.save();c.translate(340+6*q,204-1.2*q);c.rotate(-7*q*Math.PI/180);c.translate(-340,-204);c.drawImage(this.s.face,0,0);
  if(eye>.01){c.globalAlpha=eye;iris(c,'M334 127 C338 129 342 131.5 347 134.6 C344 136.2 340 135.6 337 132.2 Z',340,132,1250,515,2.2,2.5);iris(c,'M363.6 138 C367.6 139 372.8 139.7 377.4 138.8 C374.9 143.2 369 145.2 365 142 Z',370,141,1250,515,2.2,2.7)}c.restore();
 }
 // Action 6: an explicitly illustrative change in the receiver's local reading rule.
 // Both symbols keep the same red pigment; only the receiver's shape changes.
 codebookMeaning(t){const c=this.ctx,change=ramp(t,19.8,23.6);c.drawImage(this.bookBase,0,0);
  c.drawImage(this.s.red,0,0); // Sender: 17 always means the red circle.
  c.save();c.translate(1167,729);c.beginPath();
  // A sampled circular boundary becomes a five-point star, retaining watercolor grain.
  for(let i=0;i<10;i++)for(let j=0;j<8;j++){
   const f=j/8,a=-Math.PI/2+i*Math.PI/5,b=a+Math.PI/5,r0=i%2?10:24,r1=i%2?24:10;
   const sx=mix(r0*Math.cos(a),r1*Math.cos(b),f),sy=mix(r0*Math.sin(a),r1*Math.sin(b),f)*.94;
   const angle=a+f*Math.PI/5,x=mix(17.7*Math.cos(angle),sx,change),y=mix(17.1*Math.sin(angle),sy,change);
   if(i===0&&j===0)c.moveTo(x,y);else c.lineTo(x,y);
  }
  c.closePath();c.clip();c.scale(1.6,1.6);c.drawImage(this.s.red,-218,-713);c.restore();
  // These painted marks move one row toward the viewer. Enlarge around their
  // own centers, retaining the receiving page's original foreshortening and
  // pigment texture; translating alone left them undersized next to 18/19.
  drawPart(c,this.s.triangle,1148-1167,778-729,0,[1167,729],1.22);
  drawPart(c,this.s.square,1116-1148,834-778,0,[1148,778],1.22);
  const intro=fade(t,.6,1.4,3.5,4.3),success=fade(t,14.2,15,17.4,18.1),reveal=fade(t,19,19.8,24.1,24.8),final=fade(t,35.4,36.2,48,49);
  const rings=Math.max(intro,success,reveal,final);if(rings){c.save();c.globalAlpha=.7*rings;c.lineWidth=1.6;c.setLineDash([4,8]);c.strokeStyle='#b56559';c.beginPath();c.ellipse(172,723,98,44,-.18,0,Math.PI*2);c.stroke();c.strokeStyle='#b56559';c.beginPath();c.ellipse(1123,721,79,35,.22,0,Math.PI*2);c.stroke();c.restore()}
  if(reveal){c.save();c.globalAlpha=.3*reveal;c.strokeStyle='#e0b35f';c.shadowColor='#e4b77a';c.shadowBlur=8;c.lineWidth=2;c.beginPath();c.ellipse(1167,729,23+4*Math.sin(t*2),23+4*Math.sin(t*2),0,0,Math.PI*2);c.stroke();c.restore()}
 }
 // Action 7: captions explain only the currently visible beat.
 annotations(t){const c=this.ctx,x=808;
  text(c,'I · An agreed symbol codebook',x,65,34,fade(t,.2,.9,17.4,18.1));
  text(c,'17 means a circle at both ends.',x,120,27,fade(t,.8,1.4,4,4.6));
  text(c,'Claude Shannon’s problem:\nrecover the selected message.',x,121,28,fade(t,4.6,5.3,11.4,12.1));
  text(c,'The dictionary is already agreed.',x,212,23,fade(t,5.3,6,11.4,12.1));
  text(c,'17 received. Circle recovered.',x,124,29,fade(t,14,14.8,17.4,18.1));
  text(c,'II · No agreed symbol codebook',x,65,33,fade(t,18.1,18.8,39.3,40));
  text(c,'Now change the receiver’s reading rule.',x,121,25,fade(t,18.8,19.5,24.3,25));
  text(c,'Same Morse. Same numeral: 17.',x,122,28,fade(t,25,25.7,34.5,35.2));
  text(c,'Sender: circle.  Receiver: star.',x,123,29,fade(t,35.3,36,39.3,40));
  text(c,'The numeral arrived intact.\nThe interpretations differ.',x,78,33,ramp(t,40,40.8));
  text(c,'Learned representations:\na shared meaning is not guaranteed in advance.',x,205,23,ramp(t,42,42.8));
 }
 render(t){const c=this.ctx;c.save();c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,WIDTH,HEIGHT);this.drawStage();this.codebookMeaning(t);this.senderKey(t);this.receiverPaper(t);this.cableSignal(t);this.receiverGaze(t);this.senderGlance(t);this.annotations(t);c.restore()}
}
async function mount(target,options={}){const base=options.assetBase||'assets/telegram/';const load=file=>new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>rej(new Error('Cannot load '+base+file));im.src=base+file});
 const keys={painting:'painting.png',cleanMotion:'clean-motion.png',eyes:'eyes.png',cleanReceiver:'clean-receiver.png',cleanSender:'clean-sender.png'},images={};await Promise.all(Object.entries(keys).map(async([k,f])=>images[k]=await load(f)));
 target.width=WIDTH;target.height=HEIGHT;const film=new Film(target,images);queueMicrotask(()=>film.emit('loadeddata'));return film;
}
global.TelegramAnimation={mount,duration:DURATION,events:EVENTS_ALL,implementation:'Independent Canvas2D actions; no Figma timeline',handPolicy:'Two original receiver hands only; no additional arm asset'};
})(window);
