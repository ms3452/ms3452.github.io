/**
 * Static stand-in for the WebSocket backend (GitHub Pages build, no Python).
 *
 * Replays results precomputed by tools/build_static_site.py with the real BPM: for each demo
 * a state JSON plus 8-bit PNGs (one intensity map per input, the index map), and for
 * trainable demos a recorded training run (index-map keyframes, metrics, trained result).
 * It calls the same callbacks with the same messages as WaveguideWSClient, so the app does
 * not need to know which backend it talks to.
 */

/** Decode an 8-bit grayscale PNG into a Uint8Array of its pixel values (row-major). */
async function loadGray(url) {
  const blob = await (await fetch(url, { cache: 'no-cache' })).blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = document.createElement('canvas');
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const out = new Uint8Array(bmp.width * bmp.height);
  for (let i = 0; i < out.length; i++) out[i] = rgba[4 * i];
  return out;
}

export class StaticBackend {
  constructor(options = {}, base = 'data/') {
    this.onState = options.onState || (() => {});
    this.onIntensityBuffer = options.onIntensityBuffer || (() => {});
    this.onDeltaNBuffer = options.onDeltaNBuffer || (() => {});
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onTrainResult = options.onTrainResult || (() => {});
    this.base = base;
    this.cache = new Map(); // url -> promise (JSON or Uint8Array)
    this.current = null; // {key, entry, untrained: {state, intensity, dn}}
    this.playing = false;
    this.requestId = 0; // newer requests win over slower older ones
    this.init();
  }

  async init() {
    try {
      this.manifest = await this._json('manifest.json');
      this.onStatusChange(true, 'Static demo · precomputed BPM simulations');
      await this.sendDemo(this.manifest.default);
    } catch (err) {
      console.error(err);
      this.onStatusChange(false, 'Could not load the precomputed data (serve this folder over http)');
    }
  }

  _json(rel) {
    const url = this.base + rel;
    if (!this.cache.has(url)) this.cache.set(url, fetch(url, { cache: 'no-cache' }).then((r) => r.json())); // revalidate: data changes on rebuild
    return this.cache.get(url);
  }

  _gray(rel) {
    const url = this.base + rel;
    if (!this.cache.has(url)) this.cache.set(url, loadGray(url));
    return this.cache.get(url);
  }

  /** Load a state JSON and its images. */
  async _loadState(rel) {
    const state = await this._json(rel);
    const [intensity, dn] = await Promise.all([this._gray(state.files.intensity), this._gray(state.files.dn)]);
    return { state, intensity, dn };
  }

  _emit(state, intensity, dn, extra = {}) {
    this.onState({ ...state, ...extra });
    this.onIntensityBuffer(intensity);
    this.onDeltaNBuffer(dn);
  }

  async _show(entry, key) {
    const id = ++this.requestId;
    const loaded = await this._loadState(entry.state);
    if (id !== this.requestId) return; // a newer request arrived meanwhile
    this.current = { key, entry, untrained: loaded };
    this._emit(loaded.state, loaded.intensity, loaded.dn, { training: false, train_update: false });
    if (entry.train) {
      // Prefetch the recorded training so Train starts without a pause
      this._json(entry.train).then((t) => {
        this._gray(t.file);
        (t.light || []).forEach((rel) => this._gray(rel));
      });
      this._loadState(entry.trained).catch(() => {});
    }
  }

  async sendDemo(name) {
    if (this.playing || !this.manifest) return;
    const entry = this.manifest.demos[name] || this.manifest.demos[this.manifest.default];
    if (name === 'custom') {
      const perm = this.customPerm || entry.default_perm;  // after Clear: back to the default
      await this._show(this.manifest.custom[perm], 'custom');
    } else {
      await this._show(entry, name);
    }
  }

  /**
   * Routing designer (static build): only the 24 one-to-one 4x4 routings were precomputed.
   * A complete routing shows its recorded run; an incomplete one (after Clear, while the user
   * is still linking) shows the untrained device with the partial target and cannot be trained.
   */
  async sendRouting(n, connections) {
    if (this.playing || n !== 4) return;
    const perm = [null, null, null, null];
    connections.forEach(([i, o]) => {
      perm[i] = o;
    });
    const key = perm.join('');
    if (this.manifest.custom[key]) {
      this.customPerm = key;
      await this._show(this.manifest.custom[key], 'custom');
      return;
    }
    await this._showPartial(connections);
  }

  async _showPartial(links) {
    const id = ++this.requestId;
    // Every custom routing starts from the same untrained device; take any recorded one and
    // swap in the partial target. Its target_out rows are the output-mode profiles.
    const base = await this._loadState(this.manifest.custom[this.manifest.demos.custom.default_perm].state);
    if (id !== this.requestId) return;
    const d = base.state.demo;
    const profiles = [];
    d.links.forEach(([i, o]) => (profiles[o] = d.target_out[i]));
    const zeros = profiles[0].map(() => 0);
    const target = [0, 1, 2, 3].map((o) => [0, 1, 2, 3].map((i) => (links.some(([a, b]) => a === i && b === o) ? 1 : 0)));
    const targetOut = [0, 1, 2, 3].map((i) => {
      const link = links.find(([a]) => a === i);
      return link ? profiles[link[1]] : zeros;
    });
    const state = {
      ...base.state,
      train_metrics: null,
      demo: { ...d, links, target, target_out: targetOut, fidelity: null, transmission: null },
    };
    this.customPerm = null;
    this.current = { key: 'custom', entry: { state: null }, untrained: { ...base, state } };
    this._emit(state, base.intensity, base.dn, { training: false, train_update: false });
  }

  /**
   * State shown at keyframe f of a recorded run: the untrained state with that step's output
   * (profiles, detector powers -> transmission matrix, scores, MNIST confusion matrix).
   */
  _frameState(u, run, f) {
    const fs = run.states && run.states[f];
    if (!fs) return { ...u.state, train_metrics: run.metrics[f] }; // data recorded before per-step output
    const d = u.state.demo;
    return {
      ...u.state,
      max_intensity: fs.max_intensity,
      train_metrics: run.metrics[f],
      demo: {
        ...d,
        intensity_out: fs.intensity_out,
        fidelity: fs.fidelity,
        transmission: fs.transmission,
        confusion: fs.confusion,
        detector: d.detector ? { ...d.detector, powers: fs.powers } : null,
      },
    };
  }

  /** Replay the recorded training: index-map and light keyframes, then the trained result. */
  async sendTrain() {
    const cur = this.current;
    if (this.playing || !cur || !cur.entry.train) return;
    this.playing = true;
    const id = ++this.requestId;
    try {
      const run = await this._json(cur.entry.train);
      const [frames, trained, light] = await Promise.all([
        this._gray(run.file),
        this._loadState(cur.entry.trained),
        Promise.all((run.light || []).map((rel) => this._gray(rel))),
      ]);
      const u = cur.untrained;
      const frameSize = u.dn.length;
      const total = run.metrics[run.metrics.length - 1].total;
      // Start from the untrained map (also when training again)
      this._emit(u.state, u.intensity, u.dn, {
        training: true,
        train_update: true,
        train_metrics: { step: 0, total, done: false },
      });
      for (let f = 0; f < run.frames; f++) {
        await new Promise((r) => setTimeout(r, this.manifest.frame_ms));
        if (id !== this.requestId) return;
        const last = f === run.frames - 1;
        const dn = frames.subarray(f * frameSize, (f + 1) * frameSize);
        if (last) {
          this._emit(trained.state, trained.intensity, trained.dn, { training: false, train_update: true });
        } else {
          this._emit(this._frameState(u, run, f), light[f] || u.intensity, dn, {
            training: true,
            train_update: true,
          });
        }
      }
      this.onTrainResult(trained.state.train_metrics);
    } finally {
      this.playing = false;
    }
  }
}
