// Worker de detección de carriles con IA (Ultra-Fast-Lane-Detection, ResNet-18, CULane, licencia MIT).
// Corre aparte para no trabar la pantalla.
const MODEL_URL='https://huggingface.co/mlboydaisuke/Ultra-Fast-Lane-Detection-OpenVINO/resolve/main/ufld.onnx';
const ORT='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
let session=null,inName=null,nchw=true;
// filas de referencia de CULane (en una imagen de 288 de alto)
const ANCHORS=[121,131,141,150,160,170,180,189,199,209,219,228,238,248,258,267,277,287];
async function getModel(progress){
  const cache=await caches.open('carril-rapido-modelos');
  let res=await cache.match(MODEL_URL);
  if(!res){const r=await fetch(MODEL_URL);if(!r.ok)throw new Error('No se pudo descargar el modelo ('+r.status+')');
    const total=+r.headers.get('content-length')||178141391;const reader=r.body.getReader();const chunks=[];let got=0;
    for(;;){const{done,value}=await reader.read();if(done)break;chunks.push(value);got+=value.length;progress(got/total)}
    const blob=new Blob(chunks);await cache.put(MODEL_URL,new Response(blob));return new Uint8Array(await blob.arrayBuffer())}
  progress(1);return new Uint8Array(await res.arrayBuffer());
}
async function init(){
  importScripts(ORT+'ort.webgpu.min.js');ort.env.wasm.wasmPaths=ORT;ort.env.wasm.numThreads=Math.min(4,navigator.hardwareConcurrency||2);
  const bytes=await getModel(p=>postMessage({type:'progress',p}));
  const eps=('gpu' in navigator)?['webgpu','wasm']:['wasm'];
  session=await ort.InferenceSession.create(bytes,{executionProviders:eps,graphOptimizationLevel:'all'});
  inName=session.inputNames[0];
  postMessage({type:'ready',ep:eps[0]});
}
const MEAN=[0.485,0.456,0.406],STD=[0.229,0.224,0.225];
function decode(out,dims){
  // salida esperada [1,201,18,4] (cuadros+1, filas, carriles); se aceptan otros órdenes comunes
  let G=201,R=18,L=4,idx;
  if(dims[1]===201&&dims[2]===18&&dims[3]===4)idx=(g,r,l)=>(g*R+r)*L+l;
  else if(dims[1]===18&&dims[2]===4&&dims[3]===201)idx=(g,r,l)=>(r*L+l)*G+g;
  else if(dims[1]===4&&dims[2]===18&&dims[3]===201)idx=(g,r,l)=>(l*R+r)*G+g;
  else throw new Error('Salida inesperada '+dims.join('x'));
  const lanes=[];
  for(let l=0;l<L;l++){const pts=[];
    for(let r=0;r<R;r++){let mx=-1e9,arg=0;for(let g=0;g<G;g++){const v=out[idx(g,r,l)];if(v>mx){mx=v;arg=g}}
      if(arg===G-1)continue;
      let s=0,e=0;for(let g=0;g<G-1;g++){const p=Math.exp(out[idx(g,r,l)]-mx);s+=p;e+=p*(g+1)}
      const col=e/s-1; // 0..199
      pts.push([col/199,ANCHORS[r]/288])}
    lanes.push(pts.length>=4?pts:null)}
  return lanes;
}
onmessage=async e=>{const m=e.data;
  try{
    if(m.type==='init'){await init();return}
    if(m.type==='frame'&&session){
      const{data,w,h}=m; // ImageData RGBA 800x288
      const plane=w*h,inp=new Float32Array(3*plane);
      for(let i=0;i<plane;i++)for(let c=0;c<3;c++){const v=(data[i*4+c]/255-MEAN[c])/STD[c];inp[nchw?c*plane+i:i*3+c]=v}
      const t0=performance.now();
      const meta=session.inputMetadata?.[0];
      const t=new ort.Tensor('float32',inp,nchw?[1,3,h,w]:[1,h,w,3]);
      let res;try{res=await session.run({[inName]:t})}catch(err){if(nchw){nchw=false;return postMessage({type:'retry'})}throw err}
      const o=res[session.outputNames[0]];
      postMessage({type:'lanes',lanes:decode(o.data,o.dims),ms:performance.now()-t0,id:m.id});
    }
  }catch(err){postMessage({type:'error',msg:String(err.message||err)})}
};
