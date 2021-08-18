"use strict";(self["webpackChunkDANER"]=self["webpackChunkDANER"]||[]).push([[312],{7312:(e,t,l)=>{l.r(t),l.d(t,{default:()=>j});l(71);var n=l(3673),a=l(8880);const o=(0,n.HX)("data-v-0b625c80");(0,n.dD)("data-v-0b625c80");const s={class:"myContainer"},r=(0,n.Wm)("div",{class:"text-h4"},"Name Entity Recognition Service",-1),c={class:"q-ma-md"},u={class:"q-gutter-sm"},i={class:"q-pa-md"},d={class:"q-gutter-md row items-start"};(0,n.Cn)();const m=o(((e,t,l,m,p,v)=>{const g=(0,n.up)("q-card-section"),b=(0,n.up)("q-checkbox"),f=(0,n.up)("q-separator"),x=(0,n.up)("q-btn"),y=(0,n.up)("q-select"),S=(0,n.up)("q-card"),h=(0,n.up)("q-input"),L=(0,n.up)("ner-render"),W=(0,n.up)("q-page");return(0,n.wg)(),(0,n.j4)(W,{padding:""},{default:o((()=>[(0,n.Wm)("div",s,[(0,n.Wm)(S,null,{default:o((()=>[(0,n.Wm)(g,{class:"bg-primary text-white",align:"center"},{default:o((()=>[r])),_:1}),(0,n.Wm)("div",c,[(0,n.Wm)("div",u,[((0,n.wg)(!0),(0,n.j4)(n.HY,null,(0,n.Ko)(m.ALL_ENTS,((e,l)=>((0,n.wg)(),(0,n.j4)(b,{modelValue:m.ents,"onUpdate:modelValue":t[1]||(t[1]=e=>m.ents=e),"keep-color":"",key:l,val:e.val,label:e.label,color:e.color},null,8,["modelValue","val","label","color"])))),128))]),(0,n.Wm)(f),(0,n.Wm)(x,{color:"primary",label:"Select All",onClick:m.onSelectALL},null,8,["onClick"]),(0,n.Wm)(x,{color:"secondary",label:"Default",onClick:m.onSelectDefault,class:"q-ma-md"},null,8,["onClick"]),(0,n.Wm)(x,{color:"primary",label:"Reset",flat:"",onClick:m.onResetLabel,class:"q-ma-md"},null,8,["onClick"])]),(0,n.Wm)("div",i,[(0,n.Wm)("div",d,[(0,n.Wm)(y,{filled:"",modelValue:m.model,"onUpdate:modelValue":t[2]||(t[2]=e=>m.model=e),options:m.ALL_MODELS,label:"Model",style:{width:"250px"}},null,8,["modelValue","options"])])])])),_:1}),(0,n.Wm)(S,null,{default:o((()=>[(0,n.Wm)("form",{onSubmit:t[4]||(t[4]=(0,a.iM)(((...e)=>m.onSubmit&&m.onSubmit(...e)),["prevent","stop"])),onReset:t[5]||(t[5]=(0,a.iM)(((...e)=>m.onReset&&m.onReset(...e)),["prevent","stop"])),class:"q-gutter-md"},[(0,n.Wm)("div",null,[(0,n.Wm)(h,{ref:"textRef",filled:"",type:"textarea",modelValue:m.text,"onUpdate:modelValue":t[3]||(t[3]=e=>m.text=e),hint:"Max characters 1500",counter:"","lazy-rules":"",autogrow:"",rules:m.textRules,"input-style":{"min-height":"100px"}},null,8,["modelValue","rules"])]),(0,n.Wm)("div",null,[(0,n.Wm)(x,{label:"Analyze",type:"submit",color:"primary",class:"q-ma-md"}),(0,n.Wm)(x,{label:"Reset",type:"reset",color:"primary",flat:"",class:"q-ma-md"})])],32)])),_:1}),(0,n.Wm)(L,{nerText:m.text,nerSpans:m.result,ents:m.ents,entStyle:m.ALL_ENTS},null,8,["nerText","nerSpans","ents","entStyle"])])])),_:1})}));var p=l(8825),v=l(1959),g=l(7874),b=l(2323);const f=(0,n.HX)("data-v-5ef69569");(0,n.dD)("data-v-5ef69569");const x={key:0},y={key:1},S={class:"label-entity"},h={key:2};(0,n.Cn)();const L=f(((e,t,l,a,o,s)=>{const r=(0,n.up)("q-banner"),c=(0,n.up)("q-card-section"),u=(0,n.up)("q-card");return l.nerSpans.length>0?((0,n.wg)(),(0,n.j4)(u,{key:0},{default:f((()=>[(0,n.Wm)(c,null,{default:f((()=>[((0,n.wg)(!0),(0,n.j4)(n.HY,null,(0,n.Ko)(a.contentList,((e,t)=>((0,n.wg)(),(0,n.j4)("span",{key:t},["<br>"===e.text?((0,n.wg)(),(0,n.j4)("p",x)):""!==e.color?((0,n.wg)(),(0,n.j4)("span",y,[(0,n.Wm)(r,{dense:"",style:{background:a.getPaletteColor(e.color)}},{default:f((()=>[(0,n.Uk)((0,b.zw)(e["text"])+" ",1),(0,n.Wm)("span",S,(0,b.zw)(e["label"]),1)])),_:2},1032,["style"])])):((0,n.wg)(),(0,n.j4)("span",h,(0,b.zw)(e["text"]),1))])))),128))])),_:1})])),_:1})):(0,n.kq)("",!0)}));l(7280);var W=l(2156);const k={props:["nerText","nerSpans","ents","entStyle"],setup(e){const{getPaletteColor:t}=W.ZP;let l=null;const a=(0,n.Fl)((()=>{let t=[],n=0;return e.nerSpans.length>0&&(e.nerSpans.forEach((({_:a,start:o,end:s,label:r})=>{const c=e.nerText.slice(o,s),u=e.nerText.slice(n,o).split("\n");u.forEach(((e,l)=>{t.push({text:e,label:"",color:""}),u.length>1&&l!=u.length-1&&t.push({text:"<br>",label:"",color:""})})),e.ents.includes(r)?(l=e.entStyle.find((({val:e})=>e===r)).color,t.push({text:c,label:r,color:l})):t.push({text:c,label:"",color:""}),n=s})),t.push({text:e.nerText.slice(n,e.nerText.length),label:"",color:""})),t}));return{contentList:a,getPaletteColor:t}}};var q=l(151),N=l(5589),C=l(5607),_=l(7518),w=l.n(_);k.render=L,k.__scopeId="data-v-5ef69569";const R=k;w()(k,"components",{QCard:q.Z,QCardSection:N.Z,QBanner:C.Z});const E={components:{NerRender:R},setup(){const e=(0,p.Z)(),t=(0,g.oR)(),l=t.state.serviceNer.ALL_MODELS,a=t.state.serviceNer.ALL_ENTS,o=(0,v.iH)(null),s=(0,n.Fl)({get:()=>t.state.serviceNer.text,set:e=>{t.commit("serviceNer/updateText",e)}}),r=(0,n.Fl)({get:()=>t.state.serviceNer.ents,set:e=>{t.commit("serviceNer/updateEnts",e)}}),c=(0,n.Fl)({get:()=>t.state.serviceNer.model,set:e=>{t.commit("serviceNer/updateModel",e)}}),u=(0,n.Fl)((()=>t.state.serviceNer.result)),i=[e=>e&&e.length>0||"Please type something to analyze!",e=>e&&e.length<1500||"Text too long"];function d(){r.value=t.getters["serviceNer/getAllEnts"]}function m(){r.value=t.getters["serviceNer/getDefaultEnts"]}function b(){r.value=[]}function f(){o.value.validate(),o.value.hasError?e.notify({color:"negative",message:"No content to analyze"}):e.notify({icon:"done",color:"positive",message:"Send to server for analyze"}),t.dispatch("serviceNer/analyzeText",{textInput:s.value})}function x(){s.value=null,o.value.resetValidation()}return{text:s,textRef:o,textRules:i,result:u,ents:r,model:c,ALL_ENTS:a,ALL_MODELS:l,onSelectALL:d,onSelectDefault:m,onResetLabel:b,onSubmit:f,onReset:x}}};var A=l(4379),T=l(5735),Z=l(5869),D=l(8240),Q=l(5987),V=l(4842);E.render=m,E.__scopeId="data-v-0b625c80";const j=E;w()(E,"components",{QPage:A.Z,QCard:q.Z,QCardSection:N.Z,QCheckbox:T.Z,QSeparator:Z.Z,QBtn:D.Z,QSelect:Q.Z,QInput:V.Z})}}]);