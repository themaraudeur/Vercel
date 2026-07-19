// Runs the clothing-segmentation model in a background thread so the page
// (buttons, scroll, animations) never freezes while the model loads or infers.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;

let segmenter = null;

// Separable box blur = O(w*h) instead of O(w*h*radius^2), fast even on phones.
function boxFilter(arr, w, h, radius, op) {
  // op: 'avg' | 'max' | 'min'
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const passes = [
    // horizontal
    (src, dst) => {
      for (let y = 0; y < h; y++) {
        const rowOff = y * w;
        for (let x = 0; x < w; x++) {
          const x0 = Math.max(0, x - radius), x1 = Math.min(w - 1, x + radius);
          let acc = op === 'avg' ? 0 : (op === 'max' ? -Infinity : Infinity);
          let count = 0;
          for (let xx = x0; xx <= x1; xx++) {
            const v = src[rowOff + xx];
            if (op === 'avg') acc += v;
            else if (op === 'max') { if (v > acc) acc = v; }
            else { if (v < acc) acc = v; }
            count++;
          }
          dst[rowOff + x] = op === 'avg' ? acc / count : acc;
        }
      }
    },
    // vertical
    (src, dst) => {
      for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
          const y0 = Math.max(0, y - radius), y1 = Math.min(h - 1, y + radius);
          let acc = op === 'avg' ? 0 : (op === 'max' ? -Infinity : Infinity);
          let count = 0;
          for (let yy = y0; yy <= y1; yy++) {
            const v = src[yy * w + x];
            if (op === 'avg') acc += v;
            else if (op === 'max') { if (v > acc) acc = v; }
            else { if (v < acc) acc = v; }
            count++;
          }
          dst[y * w + x] = op === 'avg' ? acc / count : acc;
        }
      }
    }
  ];
  passes[0](arr, tmp);
  passes[1](tmp, out);
  return out;
}
const dilate = (arr, w, h, r) => boxFilter(arr, w, h, r, 'max');
const erode  = (arr, w, h, r) => boxFilter(arr, w, h, r, 'min');
const blur   = (arr, w, h, r) => boxFilter(arr, w, h, r, 'avg');

// Morphological closing (dilate then erode): reconnects a garment that the
// model split into several nearby fragments, and fills small interior holes
// — without expanding the overall outline the way plain dilation would.
function closeMask(arr, w, h, radius) {
  return erode(dilate(arr, w, h, radius), w, h, radius);
}

const GARMENT_LABELS = ['Hat','Upper-clothes','Skirt','Pants','Dress','Belt','Left-shoe','Right-shoe','Bag','Scarf','Sunglasses'];

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'segment') {
    const { dataUrl, width, height } = e.data;
    try {
      if (!segmenter) {
        segmenter = await pipeline('image-segmentation', 'Xenova/segformer_b2_clothes', {
          dtype: 'q8', // quantized weights: smaller download, faster inference on phones
          progress_callback: (p) => {
            if (p.status === 'progress') {
              self.postMessage({ type: 'progress', pct: Math.round(p.progress || 0) });
            } else if (p.status === 'download' || p.status === 'init') {
              self.postMessage({ type: 'stage', text: 'Téléchargement du modèle…' });
            }
          }
        });
      }

      self.postMessage({ type: 'stage', text: 'Analyse de la photo…' });
      const output = await segmenter(dataUrl);

      const results = [];
      const transfers = [];
      for (const item of output) {
        if (!GARMENT_LABELS.includes(item.label)) continue;
        const w = width, h = height;
        const maskImg = (item.mask.width === w && item.mask.height === h) ? item.mask : item.mask.resize(w, h);
        const data = maskImg.data;
        let area = 0;
        let raw = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
          const v = data[i] / 255;
          raw[i] = v > 0.5 ? 1 : 0; // binarize first so closing works on clean regions
          if (v > 0.5) area++;
        }
        if (area < w * h * 0.008) continue;

        // 1) close: reconnect fragments of the SAME garment split by shadows/folds,
        //    and fill small holes inside the shape.
        const closed = closeMask(raw, w, h, 3);
        // 2) light extra dilation so the mask reaches slightly past the model's
        //    (often conservative) outline, covering missed edges.
        const grown = dilate(closed, w, h, 1);
        // 3) smooth the final edge so the color transition looks natural.
        const finalMask = blur(grown, w, h, 3);

        results.push({ label: item.label, buffer: finalMask.buffer });
        transfers.push(finalMask.buffer);
      }

      self.postMessage({ type: 'done', results, width, height }, transfers);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message || String(err) });
    }
  }
};
