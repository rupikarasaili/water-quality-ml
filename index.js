// ====== CONFIG (edit these) ======
const MODEL_URL = 'WaterQuality_DeepMLP_Classifier.onnx';  
const FEATURE_NAMES = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine', 'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead', 'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 'selenium', 'silver', 'uranium']
const SAMPLE_ROW = [1.65, 9.08, 0.04, 2.85, 0.007, 0.35, 0.83, 0.17, 0.05, 0.2, 0, 0.054, 16.08, 1.13, 0.007, 37.75, 6.78, 0.08, 0.34, 0.02
  
];

// ====== DOM ======
const form = document.getElementById('form');
const inputsDiv = document.getElementById('inputs');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const errorEl = document.getElementById('error');
const predictBtn = document.getElementById('predictBtn');
const loadSampleBtn = document.getElementById('loadSampleBtn');
const resetBtn = document.getElementById('resetBtn');
const thrRange = document.getElementById('thr');
const thrVal = document.getElementById('thrVal');
const modelNameEl = document.getElementById('modelName');
const csvArea = document.getElementById('csvArea');
const loadCsvBtn = document.getElementById('loadCsvBtn');

let session = null;

// ====== UI builders ======
function buildInputs() {
  inputsDiv.innerHTML = '';
  FEATURE_NAMES.forEach((name, i) => {
    const wrap = document.createElement('div');

    const lab = document.createElement('label');
    lab.textContent = name;
    lab.htmlFor = `f_${name}`;

    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = 'any';
    inp.inputMode = 'decimal';
    inp.id = `f_${name}`;
    inp.placeholder = name;

    // Prefill with sample if provided
    if (SAMPLE_ROW[i] != null) inp.value = SAMPLE_ROW[i];

    wrap.appendChild(lab);
    wrap.appendChild(inp);
    inputsDiv.appendChild(wrap);
  });
}

function getVector() {
  const v = new Float32Array(FEATURE_NAMES.length);
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    const el = document.getElementById(`f_${FEATURE_NAMES[i]}`);
    const num = Number(el.value);
    if (Number.isNaN(num)) throw new Error(`Missing/invalid "${FEATURE_NAMES[i]}"`);
    v[i] = num; // NO SCALING — raw values
  }
  return v;
}

function parseCsvRow(s) {
  return s.split(/[,\s]+/).filter(Boolean).map(Number);
}

// ====== Init model ======
async function init() {
  try {
    modelNameEl.textContent = MODEL_URL;
    statusEl.textContent = 'Loading model…';

    // Build inputs now (so user can type while model loads)
    buildInputs();

    session = await ort.InferenceSession.create(
      new URL('./' + MODEL_URL, window.location.href).toString(),
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    );

    // Log available names to help if mismatched
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);

    statusEl.textContent = 'Model: ready';
  } catch (e) {
    statusEl.textContent = 'Failed to load model';
    errorEl.textContent = e.message || String(e);
    predictBtn.disabled = true;
  }
}

// ====== Predict ======
async function predict(e) {
  e.preventDefault();
  errorEl.textContent = '';
  resultEl.textContent = '';

  try {
    if (!session) throw new Error('Model not loaded');

    const thr = Number(thrRange.value || 0.5);
    thrVal.textContent = thr.toFixed(2);

    const x = getVector(); // raw vector
    const inputTensor = new ort.Tensor('float32', x, [1, x.length]);

    // Use actual names to be robust
    const inputName  = session.inputNames[0];
    const outputName = session.outputNames.includes('prob') ? 'prob' : session.outputNames[0];

    const out = await session.run({ [inputName]: inputTensor });
    let p = Number(out[outputName].data[0]);

    // If your model outputs logits (not prob), uncomment next line to apply sigmoid:
    // p = 1 / (1 + Math.exp(-p));

    const cls = p >= thr ? 1 : 0;
    resultEl.textContent = `Probability (safe=1): ${p.toFixed(4)}  →  Predicted class: ${cls}`;
  } catch (err) {
    errorEl.textContent = err.message || String(err);
  }
}

// ====== Events ======
form.addEventListener('submit', predict);
thrRange.addEventListener('input', () => thrVal.textContent = Number(thrRange.value).toFixed(2));
loadSampleBtn.addEventListener('click', () => {
  FEATURE_NAMES.forEach((n,i) => {
    const el = document.getElementById(`f_${n}`);
    if (el && SAMPLE_ROW[i] != null) el.value = SAMPLE_ROW[i];
  });
});
resetBtn.addEventListener('click', () => {
  inputsDiv.querySelectorAll('input[type="number"]').forEach(el => el.value = '');
  resultEl.textContent = ''; errorEl.textContent = '';
});
loadCsvBtn.addEventListener('click', () => {
  try {
    const vals = parseCsvRow(csvArea.value);
    if (vals.length !== FEATURE_NAMES.length) {
      throw new Error(`Expected ${FEATURE_NAMES.length} numbers, got ${vals.length}`);
    }
    FEATURE_NAMES.forEach((n,i) => {
      const el = document.getElementById(`f_${n}`);
      if (el) el.value = vals[i];
    });
    errorEl.textContent = '';
  } catch (e) {
    errorEl.textContent = e.message || String(e);
  }
});

// Boot
init();
