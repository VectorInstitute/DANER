"use strict";(self["webpackChunkDANER"]=self["webpackChunkDANER"]||[]).push([[220],{9220:(e,t,n)=>{n.r(t),n.d(t,{default:()=>J});var o=n(3673);const a=(0,o.HX)("data-v-060cdd2e");(0,o.dD)("data-v-060cdd2e");const l={class:"myContainer"};(0,o.Cn)();const r=a(((e,t,n,r,i,s)=>{const c=(0,o.up)("ner-configuration"),u=(0,o.up)("ner-annotation-render"),d=(0,o.up)("q-page");return(0,o.wg)(),(0,o.j4)(d,{padding:""},{default:a((()=>[(0,o.Wm)("div",l,[(0,o.Wm)(c),r.startAnnotation?((0,o.wg)(),(0,o.j4)(u,{key:0})):(0,o.kq)("",!0)])])),_:1})}));var i=n(8825),s=n(7874);const c=(0,o.HX)("data-v-511eeee8");(0,o.dD)("data-v-511eeee8");const u=(0,o.Wm)("div",{class:"text-h4"},"Annotation",-1),d={class:"q-gutter-sm"};(0,o.Cn)();const m=c(((e,t,n,a,l,r)=>{const i=(0,o.up)("q-card-section"),s=(0,o.up)("q-btn"),m=(0,o.up)("q-radio"),p=(0,o.up)("q-separator"),g=(0,o.up)("token-render"),f=(0,o.up)("q-card");return(0,o.wg)(),(0,o.j4)(f,null,{default:c((()=>[(0,o.Wm)(i,{class:"bg-primary text-white q-pa-sm",align:"center"},{default:c((()=>[u])),_:1}),(0,o.Wm)(i,{class:"flex justify-between"},{default:c((()=>[(0,o.Wm)("div",null,[(0,o.Wm)(s,{dense:"",round:"",color:"primary",icon:"arrow_back",class:"q-ma-sm",onClick:a.retrievePrev},null,8,["onClick"]),(0,o.Wm)(s,{dense:"",round:"",color:"primary",icon:"arrow_forward",class:"q-ma-sm",onClick:a.retrieveNext},null,8,["onClick"])]),(0,o.Wm)("div",null,[(0,o.Wm)(s,{dense:"",round:"",color:"primary",icon:"done",class:"q-ma-sm",onClick:a.onSubmitAnnotation},null,8,["onClick"]),(0,o.Wm)(s,{dense:"",round:"",color:"primary",icon:"close",class:"q-ma-sm",onClick:a.onRejectAnnotation},null,8,["onClick"])])])),_:1}),(0,o.Wm)(i,null,{default:c((()=>[(0,o.Wm)("div",d,[((0,o.wg)(!0),(0,o.j4)(o.HY,null,(0,o.Ko)(a.entList,((e,n)=>((0,o.wg)(),(0,o.j4)(m,{modelValue:a.curEnt,"onUpdate:modelValue":t[1]||(t[1]=e=>a.curEnt=e),"keep-color":"",key:n,val:e.val,label:e.label,color:e.color},null,8,["modelValue","val","label","color"])))),128))])])),_:1}),(0,o.Wm)(p),(0,o.Wm)(i,null,{default:c((()=>[((0,o.wg)(!0),(0,o.j4)(o.HY,null,(0,o.Ko)(a.contentList,((e,t)=>((0,o.wg)(),(0,o.j4)(g,{key:t,id:e.id,token:e,onMousedown:a.handleMouseDown,onMouseup:a.handleMouseUp},null,8,["id","token","onMousedown","onMouseup"])))),128))])),_:1})])),_:1})}));var p=n(2323);const g=(0,o.HX)("data-v-aab4369a");(0,o.dD)("data-v-aab4369a");const f={key:0,class:"label-entity"};(0,o.Cn)();const v=g(((e,t,n,a,l,r)=>{const i=(0,o.up)("q-btn"),s=(0,o.up)("q-banner"),c=(0,o.up)("q-tooltip");return(0,o.wg)(),(0,o.j4)("span",{id:n.id},[""!==n.token.color&&n.token.confidence>=a.conf?((0,o.wg)(),(0,o.j4)(i,{key:0,dense:"",flat:"","no-caps":"",style:{background:a.getColor(n.token)}},{default:g((()=>[(0,o.Uk)((0,p.zw)(n.token.token)+" ",1),""!==n.token.color?((0,o.wg)(),(0,o.j4)("span",f,(0,p.zw)(n.token.label),1)):(0,o.kq)("",!0)])),_:1},8,["style"])):((0,o.wg)(),(0,o.j4)(s,{key:1,dense:""},{default:g((()=>[(0,o.Uk)((0,p.zw)(n.token.token),1)])),_:1})),a.showConfidence?((0,o.wg)(),(0,o.j4)(c,{key:2,offset:[0,4]},{default:g((()=>[(0,o.Uk)((0,p.zw)(n.token.confidence.toFixed(3)),1)])),_:1})):(0,o.kq)("",!0)],8,["id"])}));var k=n(2156);const w={props:["token","id"],setup(){const e=(0,s.oR)(),{getPaletteColor:t}=k.ZP,n=(0,o.Fl)({get:()=>e.state.annotationNer.showConfidence}),a=(0,o.Fl)({get:()=>e.state.annotationNer.autoSuggestStrength/100});function l(e){return""!==e.color?t(e.color):"#fff"}return{conf:a,showConfidence:n,getColor:l}}};var N=n(8240),C=n(5607),h=n(8870),S=n(7518),y=n.n(S);w.render=v,w.__scopeId="data-v-aab4369a";const b=w;y()(w,"components",{QBtn:N.Z,QBanner:C.Z,QTooltip:h.Z});const W={components:{tokenRender:b},setup(){const e=(0,i.Z)(),t=(0,s.oR)(),n=(0,o.Fl)({get:()=>t.state.annotationNer.startIndex,set:e=>{t.commit("annotationNer/updateStartIndex",e)}}),a=(0,o.Fl)({get:()=>t.state.annotationNer.curEnt,set:e=>{t.commit("annotationNer/updateCurEnt",e)}}),l=(0,o.Fl)((()=>t.state.annotationNer.result)),r=(0,o.Fl)((()=>t.getters["annotationNer/getEntList"])),c=(0,o.Fl)((()=>t.getters["annotationNer/getContentList"]));function u(e){const t=e.currentTarget;n.value=t.id}function d(e){const o=e.currentTarget;n.value===o.id?t.commit("annotationNer/toggleLabel",o.id):t.commit("annotationNer/addLabel",{endIndex:o.id}),n.value=null,window.getSelection?window.getSelection().removeAllRanges():document.selection&&document.selection.empty()}function m(){e.notify({position:"top",spinner:!0,message:"Send to backend",timeout:1e3}),t.dispatch("annotationNer/updateDataset")}function p(){t.dispatch("annotationNer/getDataset")}function g(){t.commit("annotationNer/retrieveHistoryPrev")}function f(){t.commit("annotationNer/retrieveHistoryNext")}return{curEnt:a,result:l,entList:r,contentList:c,handleMouseDown:u,handleMouseUp:d,onSubmitAnnotation:m,onRejectAnnotation:p,retrievePrev:g,retrieveNext:f}}};var L=n(151),q=n(5589),_=n(7991),A=n(5869);W.render=m,W.__scopeId="data-v-511eeee8";const x=W;y()(W,"components",{QCard:L.Z,QCardSection:q.Z,QBtn:N.Z,QRadio:_.Z,QSeparator:A.Z});const Z=(0,o.Wm)("div",{class:"text-h4"},"Configuration",-1),V={class:"row",style:{}},E={class:"col"},Q={class:"q-gutter-sm"},j={class:"col"},F=(0,o.Uk)(" Confidence Level: 0 "),R=(0,o.Uk)(" 100 "),D={class:"q-pa-md"},U={class:"q-gutter-md row items-start"};function M(e,t,n,a,l,r){const i=(0,o.up)("q-card-section"),s=(0,o.up)("q-checkbox"),c=(0,o.up)("q-btn"),u=(0,o.up)("q-separator"),d=(0,o.up)("q-toggle"),m=(0,o.up)("q-item-section"),p=(0,o.up)("q-slider"),g=(0,o.up)("q-item"),f=(0,o.up)("q-select"),v=(0,o.up)("q-card");return(0,o.wg)(),(0,o.j4)(v,null,{default:(0,o.w5)((()=>[(0,o.Wm)(i,{class:"bg-primary text-white q-pa-sm",align:"center"},{default:(0,o.w5)((()=>[Z])),_:1}),(0,o.Wm)(i,null,{default:(0,o.w5)((()=>[(0,o.Wm)("div",V,[(0,o.Wm)("div",E,[(0,o.Wm)("div",Q,[((0,o.wg)(!0),(0,o.j4)(o.HY,null,(0,o.Ko)(a.ALL_ENTS,((e,n)=>((0,o.wg)(),(0,o.j4)(s,{modelValue:a.ents,"onUpdate:modelValue":t[1]||(t[1]=e=>a.ents=e),"keep-color":"",key:n,val:e.val,label:e.label,color:e.color},null,8,["modelValue","val","label","color"])))),128))]),(0,o.Wm)(c,{color:"primary",label:"Select All",onClick:a.onSelectALL},null,8,["onClick"]),(0,o.Wm)(c,{color:"secondary",label:"Default",onClick:a.onSelectDefault,class:"q-ma-md"},null,8,["onClick"]),(0,o.Wm)(c,{color:"primary",label:"Reset",flat:"",onClick:a.onResetLabel,class:"q-ma-md"},null,8,["onClick"])]),(0,o.Wm)(u,{vertical:""}),(0,o.Wm)("div",j,[(0,o.Wm)("div",null,[(0,o.Wm)(d,{label:"Active Learning",modelValue:a.activeLearningOn,"onUpdate:modelValue":t[2]||(t[2]=e=>a.activeLearningOn=e)},null,8,["modelValue"]),(0,o.Wm)(d,{label:"Show Confidence",modelValue:a.showConfidence,"onUpdate:modelValue":t[3]||(t[3]=e=>a.showConfidence=e)},null,8,["modelValue"]),(0,o.Wm)(d,{label:"Early Phase",modelValue:a.earlyPhaseOn,"onUpdate:modelValue":t[4]||(t[4]=e=>a.earlyPhaseOn=e)},null,8,["modelValue"])]),(0,o.Wm)("div",null,[(0,o.Wm)(g,null,{default:(0,o.w5)((()=>[(0,o.Wm)(m,{side:""},{default:(0,o.w5)((()=>[F])),_:1}),(0,o.Wm)(m,null,{default:(0,o.w5)((()=>[(0,o.Wm)(p,{modelValue:a.autoSuggestStrength,"onUpdate:modelValue":t[5]||(t[5]=e=>a.autoSuggestStrength=e),min:0,max:100,label:""},null,8,["modelValue"])])),_:1}),(0,o.Wm)(m,{side:""},{default:(0,o.w5)((()=>[R])),_:1})])),_:1})]),(0,o.Wm)("div",D,[(0,o.Wm)("div",U,[(0,o.Wm)(f,{filled:"",modelValue:a.model,"onUpdate:modelValue":t[6]||(t[6]=e=>a.model=e),options:a.ALL_MODELS,label:"Model",style:{width:"175px"}},null,8,["modelValue","options"]),(0,o.Wm)(c,{color:"primary",label:"start",onClick:a.onStartAnnotation,class:"q-ma-md"},null,8,["onClick"])])])])])])),_:1})])),_:1})}var O=n(1959);const P={setup(){const e=(0,i.Z)(),t=(0,s.oR)(),n=(0,o.Fl)({get:()=>t.state.annotationNer.earlyPhaseOn,set:e=>{t.commit("annotationNer/updateEarlyPhaseOn",e)}}),a=(0,o.Fl)({get:()=>t.state.annotationNer.activeLearningOn,set:e=>{t.commit("annotationNer/updateActiveLearningOn",e)}}),l=(0,o.Fl)({get:()=>t.state.annotationNer.showConfidence,set:e=>{t.commit("annotationNer/updateShowConfidence",e)}}),r=(0,o.Fl)({get:()=>t.state.annotationNer.autoSuggestStrength,set:e=>{t.commit("annotationNer/updateAutoSuggestStrength",e)}}),c=(0,o.Fl)({get:()=>t.state.annotationNer.tradeoff,set:e=>{t.commit("annotationNer/updateTradeoff",e)}}),u=t.state.annotationNer.ALL_MODELS,d=t.state.annotationNer.ALL_ENTS,m=(0,O.iH)(null),p=(0,o.Fl)({get:()=>t.state.annotationNer.text,set:e=>{t.commit("annotationNer/updateText",e)}}),g=(0,o.Fl)({get:()=>t.state.annotationNer.ents,set:e=>{t.commit("annotationNer/updateEnts",e)}}),f=(0,o.Fl)({get:()=>t.state.annotationNer.model,set:e=>{t.commit("annotationNer/updateModel",e)}}),v=(0,o.Fl)((()=>t.state.annotationNer.result)),k=[e=>e&&e.length>0||"Please type something to analyze!",e=>e&&e.length<1500||"Text too long"];function w(){g.value=t.getters["annotationNer/getAllEnts"]}function N(){g.value=t.getters["annotationNer/getDefaultEnts"]}function C(){g.value=[]}function h(){t.commit("annotationNer/updateStartAnnotation",!0),t.dispatch("annotationNer/getDataset")}function S(){m.value.validate(),m.value.hasError?e.notify({color:"negative",message:"No content to analyze"}):e.notify({icon:"done",color:"positive",message:"Send to server for analyze"}),t.dispatch("annotationNer/analyzeText",{textInput:p.value})}function y(){p.value=null,m.value.resetValidation()}return{earlyPhaseOn:n,activeLearningOn:a,showConfidence:l,autoSuggestStrength:r,tradeoff:c,text:p,textRef:m,textRules:k,result:v,ents:g,model:f,ALL_ENTS:d,ALL_MODELS:u,onSelectALL:w,onSelectDefault:N,onResetLabel:C,onStartAnnotation:h,onSubmit:S,onReset:y}}};var T=n(5735),H=n(8886),I=n(3414),z=n(2035),B=n(2064),K=n(5987);P.render=M;const X=P;y()(P,"components",{QCard:L.Z,QCardSection:q.Z,QCheckbox:T.Z,QBtn:N.Z,QSeparator:A.Z,QToggle:H.Z,QItem:I.Z,QItemSection:z.Z,QSlider:B.Z,QSelect:K.Z});const Y={components:{NerAnnotationRender:x,NerConfiguration:X},setup(){(0,i.Z)();const e=(0,s.oR)(),t=(0,o.Fl)((()=>e.state.annotationNer.startAnnotation));return{startAnnotation:t}}};var G=n(4379);Y.render=r,Y.__scopeId="data-v-060cdd2e";const J=Y;y()(Y,"components",{QPage:G.Z,QCard:L.Z})}}]);