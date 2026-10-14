/**
 * Main application controller for the 2D Waveguide Studio.
 * Every demo launches N inputs; the server sends one intensity map per input and the display
 * sends them through the device one light pulse at a time (CW light shows all at once).
 */

// ?v= busts copies that browsers cached before the server sent Cache-Control: no-cache
import { WebGLIntensityRenderer } from './webgl_renderer.js?v=12';
import { ProfilePlotter } from './profile_plots.js?v=12';
import { WaveguideWSClient } from './ws_client.js?v=12';
import { StaticBackend } from './static_backend.js?v=12';

// The GitHub Pages build (<html data-backend="static">) replays precomputed results instead of
// talking to the Python server; the routing designer is then limited to 4x4 one-to-one routing
const STATIC = document.documentElement.dataset.backend === 'static';

class WaveguideApp {
  constructor() {
    this.glCanvas = document.getElementById('gl-canvas');
    this.facetOverlay = document.getElementById('facet-overlay');

    this.renderer = new WebGLIntensityRenderer(this.glCanvas);
    this.inputPlot = new ProfilePlotter(document.getElementById('canvas-input-plot'), {
      color: '#00f2fe',
      fillColor: 'rgba(0, 242, 254, 0.2)',
      title: 'INPUT  I(x), z = 0',
      isInput: true,
    });
    this.outputPlot = new ProfilePlotter(document.getElementById('canvas-output-plot'), {
      color: '#4facfe',
      fillColor: 'rgba(79, 172, 254, 0.25)',
      title: 'OUTPUT  I(x), z = Lz',
      isInput: false,
    });

    const callbacks = {
      onState: (state) => this._onStateUpdate(state),
      onIntensityBuffer: (buf) => this._onIntensityBuffer(buf),
      onDeltaNBuffer: (buf) => this._onDeltaNBuffer(buf),
      onStatusChange: (online, msg) => this._onStatusChange(online, msg),
      onTrainResult: (metrics) => this._onTrainResult(metrics),
    };
    // The UI is set up first so the static backend's first (synchronous-ish) state finds it
    this.ws = null;

    this.demo = null; // active demo from the server state
    this.demoKey = null;
    this.demoVersion = 0; // bumps on every state so the facet views redraw
    this.maxIntensity = 1.0;
    this.currentLz = 5000.0;
    this.currentLx = 600.0;
    this.manualLayer = -1; // manual mode: the input sent with every pulse (-1: auto-cycle)
    this.inLayer = -1; // input carried by the pulse at the input facet (-1: CW light)
    this.outLayer = -1; // input carried by the pulse at the output facet
    this.pendingIntensity = null;
    this.isTrainUpdate = false;
    this.fpsCount = 0;
    this.lastFpsTime = performance.now();

    this._initUI();
    this._initResize();
    this._startRenderLoop();
    this.ws = STATIC ? new StaticBackend(callbacks) : new WaveguideWSClient(callbacks);
  }

  _initUI() {
    document.getElementById('demo-select').addEventListener('change', (e) => {
      this.ws.sendDemo(e.target.value);
    });

    const colormapSelect = document.getElementById('colormap-select');
    colormapSelect.value = 'cyan_gold';
    this.renderer.setColormap('cyan_gold');
    colormapSelect.addEventListener('change', (e) => this.renderer.setColormap(e.target.value));

    document.getElementById('btn-pulse').addEventListener('click', () => {
      this.renderer.setPulsed(!this.renderer.pulsed);
      document.getElementById('btn-pulse-label').textContent = this.renderer.pulsed ? 'Light: Pulsed' : 'Light: CW';
    });

    // 3D setup view (the default). Three.js is loaded on demand; if it cannot load (e.g.
    // offline) the app falls back to the 2D view, which never depends on it.
    this.scene3d = null;
    const view3d = document.getElementById('view3d');
    const btnView = document.getElementById('btn-view');
    const showView = async (show3d) => {
      if (show3d && !this.scene3d) {
        try {
          const { Scene3D } = await import('./scene3d.js?v=12');
          view3d.hidden = false;
          this.scene3d = new Scene3D(view3d, this.glCanvas);
        } catch (err) {
          console.error('3D view could not be loaded', err);
          show3d = false;
        }
      }
      view3d.hidden = !show3d;
      btnView.querySelector('span').textContent = show3d ? '2D View' : '3D View';
      btnView.title = show3d ? 'Show the 2D simulation view' : 'Show the 3D setup: projector, lens, chip and fibers';
    };
    btnView.addEventListener('click', () => showView(view3d.hidden));
    showView(true);
    document.getElementById('btn-3d-reset').addEventListener('click', () => this.scene3d && this.scene3d.resetView());

    document.getElementById('btn-fullscreen').addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });

    // Training: one click trains to completion; the server streams the index map as it morphs
    document.getElementById('btn-train').addEventListener('click', () => {
      this._setTrainingUI(true);
      this.ws.sendTrain();
    });

    // Coordinate readout under the cursor
    const wrapper = document.getElementById('canvas-wrapper');
    wrapper.addEventListener('pointermove', (e) => {
      const r = wrapper.getBoundingClientRect();
      const z = ((e.clientX - r.left) / r.width) * this.currentLz;
      const x = ((e.clientY - r.top) / r.height - 0.5) * this.currentLx; // x axis points down
      document.getElementById('hud-coords').textContent = `x: ${x.toFixed(1)} μm, z: ${z.toFixed(0)} μm`;
    });

    // Inputs: auto-cycle (4 pulses each) or manual (click an input to send only that one)
    document.getElementById('input-list').addEventListener('click', (e) => {
      const li = e.target.closest('li');
      if (!li) return;
      const i = [...li.parentNode.children].indexOf(li);
      this._setManualLayer(i === this.manualLayer ? -1 : i); // clicking it again resumes cycling
    });
    document.getElementById('btn-auto-cycle').addEventListener('click', () => this._setManualLayer(-1));

    this._initRoutingDesigner();
  }

  _setManualLayer(layer, immediate = false) {
    this.manualLayer = layer;
    this.renderer.setManualLayer(layer, immediate);
    document.getElementById('btn-auto-cycle').classList.toggle('active', layer < 0);
    document.querySelectorAll('#input-list li').forEach((li, i) => li.classList.toggle('selected', i === layer));
    document.getElementById('input-hint').textContent = layer < 0
      ? 'Each input is sent with 4 pulses in a row. Click an input to send only that one.'
      : `Manual: every new pulse carries input ${layer + 1}. Click it again or Auto-cycle to resume.`;
    this._facetKey = null;
  }

  // ------------------------------------------------------------------ server state
  _onStateUpdate(state) {
    this.renderer.updateGridSize(state.nx, state.nz, state.layers || 1);
    this.currentLz = state.lz;
    this.currentLx = state.lx;
    this.inputPlot.setLx(state.lx);
    this.outputPlot.setLx(state.lx);
    document.getElementById('dim-lz').textContent = `Length Lz: ${state.lz} μm (→)`;
    document.getElementById('dim-lx').textContent = `Width Lx: ${state.lx} μm (↕)`;

    this.demo = state.demo;
    this.maxIntensity = state.max_intensity;
    this.demoVersion++;
    if (this.demo.key !== this.demoKey) {
      this.demoKey = this.demo.key;
      this._showDemo();
    }
    if (this.demo.key === 'custom') {
      this.routingN = this.demo.n_in;
      this.routingLinks = this.demo.links || [];
      document.querySelectorAll('.routing-n').forEach((b) => {
        b.classList.toggle('active', parseInt(b.dataset.n, 10) === this.routingN);
      });
      document.getElementById('demo-title').textContent = this.demo.title;
      this._drawRouting();
    }

    // Training progress (also shown on screens that did not start the training)
    this.isTrainUpdate = !!state.train_update;
    this._setTrainingUI(!!state.training, state.train_metrics);
    this._updateTrainingMetrics(state.train_metrics);
  }

  /** Static parts of the panel for a newly selected demo. */
  _showDemo() {
    const d = this.demo;
    document.getElementById('demo-select').value = d.key;
    document.getElementById('demo-title').textContent = d.title;
    document.getElementById('demo-description').textContent = d.description;
    document.getElementById('input-list').innerHTML = d.input_labels.map((l) => `<li>${l}</li>`).join('');
    this._setManualLayer(-1, true); // a new demo starts cycling through its inputs

    const show = (id, on) => {
      document.getElementById(id).style.display = on ? '' : 'none'; // '' = stylesheet default
    };
    show('output-card', !!d.detector);
    show('matrix-section', !!d.target);
    show('routing-card', d.key === 'custom');
    show('training-card', !!d.trainable);
    show('mnist-preview', d.key === 'mnist');

    const routing = d.trainable === 'routing';
    document.getElementById('train-acc-lbl').textContent = routing ? 'Fidelity' : 'Batch Acc';
    document.getElementById('train-test-lbl').textContent = routing ? 'Transmission' : 'Test Acc';
    document.getElementById('train-desc').innerHTML = routing
      ? `Optimizes the trainable region so each input couples into its target Gaussian output mode(s) (w&#8320; = ${d.out_w0}&nbsp;&mu;m, mode overlap), backpropagating through the BPM for all ${d.n_in} inputs at once. The written map is blurred by the 6&nbsp;&mu;m FWHM write kernel.`
      : 'Optimizes &Delta;n on 60k MNIST images by backpropagating through the BPM, starting from the current map. The written map is blurred by the 6&nbsp;&mu;m FWHM write kernel.';
    if (STATIC) {
      document.getElementById('train-desc').innerHTML +=
        ' <em>This page replays a training run recorded with the Python BPM.</em>';
    }

    let note = '';
    if (d.detector && d.detector.kind === 'modes') {
      note = 'Share of the input power coupled into each output mode (mode overlap).';
    } else if (d.detector) {
      note = 'Share of the detected light in each digit bin; the brightest bin is the prediction.';
    }
    document.getElementById('output-note').textContent = note;
    this._applyRoutingVisibility();
  }

  _applyRoutingVisibility() {
    if (this.demo && this.demo.key === 'custom') this._drawRouting();
  }

  // ------------------------------------------------------------------ per-pulse views
  /**
   * The facet plots, output readout, detector bins, input list and matrix highlight follow
   * the pulse currently at each facet (pulsed light) or show all inputs together (CW).
   */
  _updateFacets() {
    const d = this.demo;
    if (!d) return;
    const pulsed = this.renderer.pulsed;
    // CW light: all inputs at once, or the selected one in manual mode
    const inL = pulsed ? this.renderer.pulseLayerAt(0) : this.manualLayer;
    const outL = pulsed ? this.renderer.pulseLayerAt(1) : this.manualLayer;
    const key = `${inL}|${outL}|${this.demoVersion}`;
    if (key === this._facetKey) return;
    this._facetKey = key;
    this.inLayer = inL;
    this.outLayer = outL;

    const sum = (rows) => rows[0].map((_, i) => rows.reduce((s, r) => s + r[i], 0));
    this.inputPlot.setData(inL >= 0 ? d.intensity_in[inL] : sum(d.intensity_in), this.maxIntensity);
    this.outputPlot.setData(outL >= 0 ? d.intensity_out[outL] : sum(d.intensity_out), this.maxIntensity);
    this.outputPlot.setTarget(d.target_out ? (outL >= 0 ? d.target_out[outL] : sum(d.target_out)) : null);

    // MNIST: the entering digit's pixels next to the input profile, and its image
    if (d.mnist_pixels && inL >= 0) {
      this.inputPlot.setPixels({ values: d.mnist_pixels[inL], xMin: d.mnist_input_x[0], xMax: d.mnist_input_x[1] });
      this._drawMNISTImage(d.mnist_images[inL]);
      document.getElementById('mnist-preview-digit').textContent = inL;
    } else {
      this.inputPlot.setPixels(null);
    }

    document.querySelectorAll('#input-list li').forEach((li, i) => li.classList.toggle('active', i === inL));
    const caption = document.getElementById('pulse-caption');
    if (!pulsed) {
      caption.textContent = inL >= 0
        ? `CW light: ${d.input_labels[inL]}`
        : `CW light: all ${d.n_in} input${d.n_in > 1 ? 's' : ''} at once`;
    } else {
      caption.textContent = `Entering: ${d.input_labels[inL]}   ·   At output: ${d.input_labels[outL]}`;
    }

    this._drawOutputReadout();
    this._drawDetectorBins();
    if (d.target) {
      this._drawMatrix(document.getElementById('task-target-matrix'), d.target, outL);
      const P = d.detector.powers;
      const sim = d.target.map((row, o) => row.map((_, i) => P[i][o]));
      this._drawMatrix(document.getElementById('task-sim-matrix'), sim, outL);
    }
  }

  /** Detector powers for the pulse at the output (CW: mean over the inputs). */
  _outputPowers() {
    const P = this.demo.detector.powers;
    const L = this.outLayer;
    return L >= 0 ? P[L] : P[0].map((_, o) => P.reduce((s, row) => s + row[o], 0) / P.length);
  }

  _drawOutputReadout() {
    const d = this.demo;
    if (!d.detector) return;
    const powers = this._outputPowers();
    const labels = d.detector.labels;
    const L = this.outLayer;
    const best = powers.indexOf(Math.max(...powers));
    const route = document.getElementById('output-route');
    const title = document.getElementById('output-title');
    if (d.key === 'mnist') {
      title.textContent = L >= 0 ? `Digit ${L} → predicted:` : 'Predicted digit:';
      route.textContent = L >= 0 ? labels[best] : '–';
      route.classList.toggle('wrong', L >= 0 && best !== L);
    } else if (d.trainable === 'routing') {
      title.textContent = 'Output';
      route.textContent = L >= 0 ? `in ${L + 1} → out ${best + 1}` : 'all inputs (CW)';
      route.classList.remove('wrong');
    } else {
      title.textContent = 'Output';
      route.textContent = L >= 0 ? d.input_labels[L] : 'all inputs (CW)';
      route.classList.remove('wrong');
    }
    document.getElementById('output-bars').innerHTML = powers
      .map((p, i) => `
        <div class="bin-row">
          <span class="bin-label wide">${labels[i]}</span>
          <div class="bin-bar-bg"><div class="bin-bar-fill" style="width: ${(Math.min(1, p) * 100).toFixed(1)}%;"></div></div>
          <span class="bin-pct">${(p * 100).toFixed(0)}%</span>
        </div>`)
      .join('');
  }

  /** Output bins in the output strip and at the output facet, plus the fixed-region marks. */
  _drawDetectorBins() {
    const d = this.demo;
    const det = d && d.detector;
    const edges = det && det.edges;
    let powers = null;
    let winner = -1;
    if (edges) {
      powers = this._outputPowers();
      // Highlight the brightest bin only where it means something (prediction or routing),
      // and not for CW light where all inputs arrive at once
      if (this.outLayer >= 0 && d.trainable) winner = powers.indexOf(Math.max(...powers));
    }
    this.outputPlot.setBins(edges ? { edges, powers, winner, labelOffset: d.key === 'mnist' ? 0 : 1 } : null);

    const cv = this.facetOverlay;
    if (cv.width !== cv.clientWidth || cv.height !== cv.clientHeight) {
      cv.width = cv.clientWidth;
      cv.height = cv.clientHeight;
    }
    const ctx = cv.getContext('2d');
    const w = cv.width;
    const h = cv.height;
    ctx.clearRect(0, 0, w, h);
    if (!d) return;

    if (d.trainable_z) {
      // Mark the fixed input/output waveguide sections (not trainable)
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.45)';
      ctx.lineWidth = 1;
      ctx.font = '600 10px Inter, sans-serif';
      ctx.fillStyle = 'rgba(226, 232, 240, 0.7)';
      ctx.textBaseline = 'top';
      const zs = d.trainable_z.map((z) => (z / this.currentLz) * w);
      zs.forEach((zx) => {
        ctx.beginPath();
        ctx.moveTo(zx, 0);
        ctx.lineTo(zx, h);
        ctx.stroke();
      });
      ctx.textAlign = 'center';
      ctx.fillText('fixed input waveguides', zs[0] / 2, 6);
      ctx.fillText('trainable region', (zs[0] + zs[1]) / 2, 6);
      ctx.fillText('fixed output waveguides', (zs[1] + w) / 2, 6);
      ctx.restore();
    }
    if (!edges) return;

    const yOf = (x) => h * (0.5 + x / this.currentLx); // x axis points down
    const barW = 8;
    for (let i = 0; i < edges.length - 1; i++) {
      const yTop = yOf(edges[i]);
      const yBot = yOf(edges[i + 1]);
      if (i === winner) {
        // Glow band reaching back into the device so the winning bin stands out
        const grad = ctx.createLinearGradient(w - 90, 0, w, 0);
        grad.addColorStop(0, 'rgba(250, 204, 21, 0)');
        grad.addColorStop(1, 'rgba(250, 204, 21, 0.35)');
        ctx.fillStyle = grad;
        ctx.fillRect(w - 90, yTop, 90, yBot - yTop);
      }
      ctx.fillStyle = i === winner ? '#facc15' : (i % 2 ? 'rgba(148, 163, 184, 0.55)' : 'rgba(148, 163, 184, 0.3)');
      ctx.fillRect(w - barW, yTop + 1, barW, yBot - yTop - 2);
    }
  }

  // ------------------------------------------------------------------ training
  _onTrainResult(metrics) {
    this._setTrainingUI(false, metrics);
    this._updateTrainingMetrics(metrics);
  }

  _setTrainingUI(training, metrics = null) {
    const badge = document.getElementById('train-status-badge');
    badge.textContent = training ? 'Training...' : (metrics && metrics.done ? 'Trained' : 'Ready');
    badge.classList.toggle('training', training);
    document.getElementById('btn-train').disabled = training || !this._canTrain();
    document.getElementById('demo-select').disabled = training;
    document.getElementById('routing-canvas').style.pointerEvents = training ? 'none' : '';
    this.renderer.setIndexView(training);
  }

  /** Static build: a custom routing can only be trained once all four inputs are linked. */
  _canTrain() {
    const d = this.demo;
    if (!STATIC || !d || d.key !== 'custom') return true;
    return (d.links || []).length === 4;
  }

  _updateTrainingMetrics(metrics) {
    const d = this.demo;
    const routing = d && d.trainable === 'routing';
    const m = metrics || {};
    document.getElementById('train-step-val').textContent = m.step != null ? `${m.step}/${m.total}` : '--';
    // Routing demos report fidelity and transmission (live values when not training)
    const a = routing ? (m.fidelity != null ? m.fidelity : d.fidelity) : m.accuracy;
    const b = routing ? (m.transmission != null ? m.transmission : d.transmission) : m.test_accuracy;
    document.getElementById('train-acc-val').textContent = a != null ? `${a.toFixed(routing ? 1 : 0)}%` : '--%';
    document.getElementById('train-test-val').textContent = b != null ? `${b.toFixed(1)}%` : '--%';
    document.getElementById('train-progress-fill').style.width = m.step != null ? `${(100 * m.step) / m.total}%` : '0%';
  }

  // ------------------------------------------------------------------ buffers
  _onIntensityBuffer(uint8Array) {
    // Hold the frame until the matching index map arrives
    this.pendingIntensity = uint8Array;
  }

  _onDeltaNBuffer(uint8Array) {
    const intensity = this.pendingIntensity;
    this.pendingIntensity = null;
    if (intensity && this.isTrainUpdate) {
      // Training frame: cross-fade index map and light
      this.renderer.morphTo(intensity, uint8Array, 250, this.maxIntensity);
      return;
    }
    this.renderer.uploadDeltaNBuffer(uint8Array);
    if (intensity) this.renderer.enqueueWavefront(intensity, 0.0, null, this.maxIntensity);
  }

  _onStatusChange(online, message) {
    document.querySelector('.status-indicator').className = `status-indicator ${online ? 'online' : 'offline'}`;
    document.getElementById('status-text').textContent = message;
  }

  _initResize() {
    const handleResize = () => {
      this.inputPlot.resize();
      this.outputPlot.resize();
      this._facetKey = null; // redraw the facet views at the new size
      this._drawRouting();
    };
    window.addEventListener('resize', handleResize);
    if (window.ResizeObserver) {
      new ResizeObserver(handleResize).observe(document.getElementById('canvas-wrapper'));
    }
    setTimeout(handleResize, 100);
  }

  _startRenderLoop() {
    let lastTime = performance.now();
    const frame = (now) => {
      this.fpsCount++;
      if (now - this.lastFpsTime >= 1000) {
        document.getElementById('hud-fps').textContent = `FPS: ${this.fpsCount}`;
        this.fpsCount = 0;
        this.lastFpsTime = now;
      }
      const dt = Math.min(50.0, now - lastTime);
      lastTime = now;
      this.renderer.update(dt);
      this._updateFacets();
      this.renderer.render();
      // Same frame as the 2D draw, so the 3D chip texture reads a complete canvas
      if (this.scene3d && !document.getElementById('view3d').hidden) this.scene3d.render(this);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------ routing designer, matrices
  _initRoutingDesigner() {
    this.routingN = 4;
    this.routingLinks = [];
    this.routingDrag = null;
    const cv = document.getElementById('routing-canvas');
    const send = (links) => this.ws.sendRouting(this.routingN, links);
    if (STATIC) {
      // Only the 24 one-to-one 4x4 routings were precomputed
      document.querySelector('.routing-sizes').style.display = 'none';
      document.getElementById('routing-help').textContent =
        'Drag an input (left) onto an output (right); each input goes to exactly one output and each output ' +
        'takes one input. With all four linked, dragging swaps two routes. Clear removes all links. ' +
        'Once all four inputs are linked, press Train to replay the precomputed inverse design.';
    }

    document.querySelectorAll('.routing-n').forEach((b) => {
      b.addEventListener('click', () => {
        const n = parseInt(b.dataset.n, 10);
        this.routingN = n;
        // New size: start from the reversed order as an example
        this.ws.sendRouting(n, Array.from({ length: n }, (_, i) => [i, n - 1 - i]));
      });
    });
    document.getElementById('btn-routing-clear').addEventListener('click', () => send([]));

    const local = (e) => {
      const r = cv.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * cv.width, y: ((e.clientY - r.top) / r.height) * cv.height };
    };
    cv.addEventListener('pointerdown', (e) => {
      const p = local(e);
      const port = this._routingPortAt(p);
      if (port) {
        this.routingDrag = { from: port, x: p.x, y: p.y };
        cv.setPointerCapture(e.pointerId);
      } else {
        const hit = this._routingLinkAt(p);
        // Static build: links cannot be removed (the routing must stay one-to-one)
        if (hit >= 0 && !STATIC) send(this.routingLinks.filter((_, k) => k !== hit));
      }
      this._drawRouting();
    });
    cv.addEventListener('pointermove', (e) => {
      if (!this.routingDrag) return;
      Object.assign(this.routingDrag, local(e));
      this._drawRouting();
    });
    cv.addEventListener('pointerup', (e) => {
      const drag = this.routingDrag;
      this.routingDrag = null;
      if (drag) {
        const to = this._routingPortAt(local(e));
        if (to && to.side !== drag.from.side) {
          const [i, o] = drag.from.side === 'in' ? [drag.from.idx, to.idx] : [to.idx, drag.from.idx];
          if (STATIC) {
            if (this.routingLinks.length === this.routingN) {
              // Complete routing: i takes output o, whoever had o gets i's old output
              const perm = Array.from({ length: this.routingN }, (_, k) => k);
              this.routingLinks.forEach(([a, b]) => (perm[a] = b));
              const j = perm.indexOf(o);
              if (j !== i) {
                perm[j] = perm[i];
                perm[i] = o;
                send(perm.map((b, a) => [a, b]));
              }
            } else {
              // Building a routing after Clear: stay one-to-one by replacing links on i or o
              const links = this.routingLinks.filter(([a, b]) => a !== i && b !== o);
              send([...links, [i, o]]);
            }
          } else if (!this.routingLinks.some(([a, b]) => a === i && b === o)) {
            send([...this.routingLinks, [i, o]]);
          }
        }
      }
      this._drawRouting();
    });
  }

  _routingGeom() {
    const cv = document.getElementById('routing-canvas');
    const w = cv.clientWidth;
    const h = Math.round(w * 1.05);
    if (cv.width !== w || cv.height !== h) {
      cv.width = w;
      cv.height = h;
    }
    const n = this.routingN;
    const pad = 22;
    const y = (k) => 14 + ((k + 0.5) * (h - 28)) / n;
    return { cv, w, h, n, xin: pad, xout: w - pad, y };
  }

  _routingPortAt(p) {
    const g = this._routingGeom();
    for (let k = 0; k < g.n; k++) {
      if (Math.hypot(p.x - g.xin, p.y - g.y(k)) < 14) return { side: 'in', idx: k };
      if (Math.hypot(p.x - g.xout, p.y - g.y(k)) < 14) return { side: 'out', idx: k };
    }
    return null;
  }

  _routingCurve(g, i, o, t) {
    // Cubic Bezier with horizontal tangents at both ports (like the paper's schematics)
    const x0 = g.xin, x1 = g.xout, y0 = g.y(i), y1 = g.y(o);
    const cx = (x0 + x1) / 2;
    const u = 1 - t;
    return {
      x: u * u * u * x0 + 3 * u * u * t * cx + 3 * u * t * t * cx + t * t * t * x1,
      y: u * u * u * y0 + 3 * u * u * t * y0 + 3 * u * t * t * y1 + t * t * t * y1,
    };
  }

  _routingLinkAt(p) {
    const g = this._routingGeom();
    let best = -1;
    let bestD = 7;
    this.routingLinks.forEach(([i, o], k) => {
      for (let s = 0; s <= 40; s++) {
        const q = this._routingCurve(g, i, o, s / 40);
        const d = Math.hypot(q.x - p.x, q.y - p.y);
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      }
    });
    return best;
  }

  _drawRouting() {
    const card = document.getElementById('routing-card');
    if (!card || card.style.display === 'none') return;
    const g = this._routingGeom();
    const ctx = g.cv.getContext('2d');
    const colors = ['#ec4899', '#10b981', '#f59e0b', '#3b82f6', '#a855f7', '#ef4444', '#14b8a6', '#eab308'];
    ctx.clearRect(0, 0, g.w, g.h);
    ctx.lineCap = 'round';
    this.routingLinks.forEach(([i, o]) => {
      ctx.beginPath();
      for (let s = 0; s <= 40; s++) {
        const q = this._routingCurve(g, i, o, s / 40);
        if (s === 0) ctx.moveTo(q.x, q.y);
        else ctx.lineTo(q.x, q.y);
      }
      ctx.strokeStyle = colors[i % colors.length];
      ctx.lineWidth = 5;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    if (this.routingDrag) {
      const a = this.routingDrag.from;
      ctx.beginPath();
      ctx.moveTo(a.side === 'in' ? g.xin : g.xout, g.y(a.idx));
      ctx.lineTo(this.routingDrag.x, this.routingDrag.y);
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.font = '600 11px JetBrains Mono, monospace';
    ctx.textBaseline = 'middle';
    for (let k = 0; k < g.n; k++) {
      [[g.xin, 'right', -12], [g.xout, 'left', 12]].forEach(([x, align, dx]) => {
        ctx.beginPath();
        ctx.arc(x, g.y(k), 7, 0, Math.PI * 2);
        ctx.fillStyle = x === g.xin ? colors[k % colors.length] : '#e2e8f0';
        ctx.fill();
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = align;
        ctx.fillText(String(k + 1), x + dx, g.y(k));
      });
    }
  }

  _drawMatrix(canvas, M, col) {
    if (!canvas) return;
    const w = canvas.clientWidth;
    if (canvas.width !== w || canvas.height !== w) {
      canvas.width = w;
      canvas.height = w;
    }
    const ctx = canvas.getContext('2d');
    const n = M.length;
    const c = w / n;
    ctx.clearRect(0, 0, w, w);
    ctx.font = `600 ${n > 4 ? 8 : 11}px JetBrains Mono, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let o = 0; o < n; o++) {
      for (let i = 0; i < n; i++) {
        const v = Math.max(0, Math.min(1, M[o][i]));
        const g = Math.sqrt(v); // make weak entries visible
        ctx.fillStyle = `rgb(${Math.round(15 + 41 * g)}, ${Math.round(23 + 166 * g)}, ${Math.round(42 + 206 * g)})`;
        ctx.fillRect(i * c + 0.5, o * c + 0.5, c - 1, c - 1);
        if (n <= 4) {
          // 8x8 cells are too small for readable numbers; the shading carries it
          ctx.fillStyle = g > 0.6 ? '#020617' : '#cbd5e1';
          ctx.fillText(String(Math.round(v * 100)), i * c + c / 2, o * c + c / 2 + 0.5);
        }
      }
    }
    if (col >= 0) {
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      ctx.strokeRect(col * c + 1, 1, c - 2, w - 2);
    }
  }

  _drawMNISTImage(imgData) {
    const canvas = document.getElementById('mnist-digit-canvas');
    if (!canvas || !imgData || imgData.length < 784) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const pixelSize = w / 28;
    ctx.clearRect(0, 0, w, h);
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const val = Math.round(Math.min(1.0, Math.max(0.0, imgData[r * 28 + c])) * 255);
        ctx.fillStyle = `rgb(${val}, ${val}, ${val})`;
        ctx.fillRect(c * pixelSize, r * pixelSize, pixelSize, pixelSize);
      }
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new WaveguideApp();
});
