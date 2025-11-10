(async () => {
  const N = 20;
  const modelPath = './model.onnx';
  const status = document.getElementById('status');

  const inputsDiv = document.getElementById('inputs');
  const fields = [];
  for (let i = 0; i < N; i++) {
    const el = document.createElement('input');
    el.type = 'number'; el.step = 'any'; el.placeholder = `x${i+1}`; el.value = '0';
    inputsDiv.appendChild(el); fields.push(el);
  }

  let session;
  try {
    session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
    status.textContent = 'Model loaded ✔';
  } catch (e) { status.textContent = 'Failed to load model'; console.error(e); return; }

  const softmax2 = (a,b) => {
    const m = Math.max(a,b), ea=Math.exp(a-m), eb=Math.exp(b-m); const s=ea+eb; return [ea/s, eb/s];
  };

  async function predict() {
    const row = new Float32Array(N);
    for (let i=0;i<N;i++){ const v=parseFloat(fields[i].value); row[i]=Number.isFinite(v)?v:0; }
    const input = new ort.Tensor('float32', row, [1,N]);
    const out = await session.run({ input1: input });
    const logits = out.output1.data;                 // adjust keys if your ONNX names differ
    const [p0, p1] = softmax2(logits[0], logits[1]);
    document.getElementById('predClass').textContent = p1>=0.5 ? '1' : '0';
    document.getElementById('prob1').textContent     = p1.toFixed(4);
    document.getElementById('p0').style.width = (p0*100).toFixed(1)+'%';
    document.getElementById('p1').style.width = (p1*100).toFixed(1)+'%';
  }

  document.getElementById('predict').onclick = predict;
  document.getElementById('reset').onclick   = () => fields.forEach(f => f.value='0');
  document.getElementById('loadCsv').onclick = () => {
    const raw = document.getElementById('csv').value.trim();
    const parts = raw.split(/[,;\s]+/).filter(Boolean).slice(0,N);
    for(let i=0;i<Math.min(parts.length,N);i++) fields[i].value = parts[i];
  };
})();
