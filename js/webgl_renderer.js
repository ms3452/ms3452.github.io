/**
 * WebGL 2.0 / WebGL 1.0 / 2D Canvas Fallback Renderer
 * with Continuous Multi-Wavefront Buffer Engine.
 * 
 * Guarantees smooth, non-abrupt physical propagation of light across arbitrary
 * successive simulation updates, queuing and propagating multiple wavefronts
 * simultaneously without jumping.
 */

export class WebGLIntensityRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.useWebGL = true;

    // Try WebGL2 first, then WebGL1
    this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    this.isWebGL2 = !!(this.gl && window.WebGL2RenderingContext && this.gl instanceof WebGL2RenderingContext);

    if (!this.gl) {
      console.warn('WebGL not available, falling back to 2D Canvas rendering.');
      this.useWebGL = false;
      this.ctx2d = canvas.getContext('2d');
    }

    this.nx = 600;
    this.nz = 250;
    this.colormapId = 2; // 0: inferno, 1: viridis, 2: cyan_gold, 3: green_laser

    // Multi-Wavefront Continuous Display Buffer
    // displayIntensity stores the active composite frame currently visible on screen
    // Demos stack one intensity map per input ("layer", rows L*nx..(L+1)*nx-1); pulse k
    // carries layer k mod layers, CW light shows the sum of all layers
    this.layers = 1;
    this.displayIntensity = new Uint8Array(this.nx * this.nz);
    this.columnVersions = new Uint32Array(this.nz); // highest updateId applied to each column
    // Each server frame is normalized by its own global max. displayMax is the max that
    // everything on screen is expressed in (the newest frame's); older content is rescaled
    // to it so columns from different frames never show a brightness step.
    this.displayMax = null;
    this.activeWavefronts = [];
    this.currentUpdateId = 0;
    this.speedColsPerMs = this.nz / 1400.0; // constant physical propagation velocity
    this.onWavefrontReachEnd = null; // callback(meta) when a wave reaches the output facet

    // Pulsed-light display: the CW intensity pattern is gated by a train of Gaussian
    // envelopes travelling along z (narrowband pulse, dispersion neglected).
    this.pulsed = true;
    this.pulseTransitSec = 4.0; // time for a pulse to cross z = 0 -> Lz
    this.pulsePeriodSec = 2.0;  // launch interval -> two pulses in the device at once
    this.pulseFwhmSec = 0.4;    // temporal FWHM -> spatial FWHM = 10% of Lz
    this.pulsesPerInput = 4;    // auto mode: each input is sent with this many pulses in a row
    this.timeOrigin = performance.now();
    // Which input each pulse k carries. A rule is a fixed layer (manual mode) or -1 (auto:
    // floor(k / pulsesPerInput) mod layers). Switching applies the new rule only to pulses
    // launched after the switch, so pulses already in the device keep their input.
    this.switchPulse = -1e9;
    this.oldRule = -1;
    this.newRule = -1;

    // Index view (0..1): dims the light and shows delta-n brightly, eased toward target.
    // Used while training so the index map can be seen morphing.
    this.indexView = 0.0;
    this.indexViewTarget = 0.0;

    // Cached buffers
    this.latestIntensity = null;
    this.latestDeltaN = null;
    this.textureNeedsUpload = false;

    if (this.useWebGL) {
      this._initShaders();
      this._initBuffers();
      this._initTextures();
    }
  }

  _initShaders() {
    const gl = this.gl;

    const vsSource = `
      attribute vec2 a_position;
      varying vec2 v_texCoord;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        // v_texCoord.x: 0 (left, z=0) to 1 (right, z=Lz)
        // v_texCoord.y: 0 (bottom) to 1 (top); the fragment shader flips it so that
        // x = -Lx/2 is at the top and x = +Lx/2 at the bottom (MNIST bin 0 on top)
        v_texCoord = (a_position + 1.0) * 0.5;
      }
    `;

    const fsSource = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_intensityTex;
      uniform sampler2D u_deltaNTex;
      uniform int u_colormap;
      uniform float u_pulsed;   // 1.0 = pulse train, 0.0 = continuous wave
      uniform float u_time;     // seconds since page load
      uniform float u_repeat;   // auto mode: pulses per input
      uniform float u_switchK;  // first pulse that follows u_newRule
      uniform float u_oldRule;  // fixed layer, or -1 for auto
      uniform float u_newRule;
      uniform float u_transit;
      uniform float u_period;
      uniform float u_fwhm;
      uniform float u_indexView;
      uniform float u_layers;

      float layerOf(float k) {
        float rule = k >= u_switchK ? u_newRule : u_oldRule;
        return rule >= 0.0 ? rule : mod(floor(k / u_repeat), u_layers);
      }

      float sampleLayer(float layer, vec2 tc) {
        return texture2D(u_intensityTex, vec2(tc.x, (layer + tc.y) / u_layers)).r;
      }

      // Colormaps
      vec3 colormapInferno(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.001, 0.001, 0.013);
        vec3 c1 = vec3(0.337, 0.070, 0.447);
        vec3 c2 = vec3(0.731, 0.223, 0.337);
        vec3 c3 = vec3(0.978, 0.558, 0.035);
        vec3 c4 = vec3(0.988, 1.000, 0.643);
        if (t < 0.25) return mix(c0, c1, t * 4.0);
        if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
        if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
        return mix(c3, c4, (t - 0.75) * 4.0);
      }

      vec3 colormapViridis(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267, 0.004, 0.329);
        vec3 c1 = vec3(0.191, 0.407, 0.556);
        vec3 c2 = vec3(0.128, 0.567, 0.551);
        vec3 c3 = vec3(0.369, 0.789, 0.383);
        vec3 c4 = vec3(0.993, 0.906, 0.144);
        if (t < 0.25) return mix(c0, c1, t * 4.0);
        if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
        if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
        return mix(c3, c4, (t - 0.75) * 4.0);
      }

      vec3 colormapCyanGold(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.02, 0.05, 0.12);
        vec3 c1 = vec3(0.0, 0.75, 0.95);
        vec3 c2 = vec3(0.95, 0.85, 0.2);
        vec3 c3 = vec3(1.0, 1.0, 1.0);
        if (t < 0.4) return mix(c0, c1, t * 2.5);
        if (t < 0.8) return mix(c1, c2, (t - 0.4) * 2.5);
        return mix(c2, c3, (t - 0.8) * 5.0);
      }

      vec3 colormapGreenLaser(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.0, 0.02, 0.0);
        vec3 c1 = vec3(0.05, 0.45, 0.15);
        vec3 c2 = vec3(0.2, 0.95, 0.3);
        vec3 c3 = vec3(0.9, 1.0, 0.8);
        if (t < 0.35) return mix(c0, c1, t / 0.35);
        if (t < 0.8) return mix(c1, c2, (t - 0.35) / 0.45);
        return mix(c2, c3, (t - 0.8) / 0.2);
      }

      vec3 colormapIronbow(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.45, 0.05, 0.60);  // violet
        vec3 c1 = vec3(0.85, 0.10, 0.45);  // magenta
        vec3 c2 = vec3(1.00, 0.35, 0.05);  // orange-red
        vec3 c3 = vec3(1.00, 0.72, 0.00);  // amber
        vec3 c4 = vec3(1.00, 1.00, 0.85);  // near white
        if (t < 0.15) return mix(c0, c1, t / 0.15);
        if (t < 0.35) return mix(c1, c2, (t - 0.15) / 0.20);
        if (t < 0.65) return mix(c2, c3, (t - 0.35) / 0.30);
        return mix(c3, c4, (t - 0.65) / 0.35);
      }

      void main() {
        vec2 tc = vec2(v_texCoord.x, 1.0 - v_texCoord.y);  // texture row 0 (x = -Lx/2) at top
        float intensity = 0.0;
        if (u_pulsed > 0.5) {
          // Retarded time at this z; k = index of the nearest pulse
          float tau = u_time - v_texCoord.x * u_transit;
          float k = floor(tau / u_period + 0.5);
          float d = tau - k * u_period;
          intensity = sampleLayer(layerOf(k), tc) * exp(-2.7725887 * d * d / (u_fwhm * u_fwhm));
        } else if (u_newRule >= 0.0) {
          intensity = sampleLayer(u_newRule, tc);  // manual mode: the selected input only
        } else {
          for (int i = 0; i < 10; i++) {  // up to 10 inputs (MNIST digits)
            if (float(i) >= u_layers) break;
            intensity += sampleLayer(float(i), tc);
          }
          intensity = min(intensity, 1.0);
        }
        float deltaN = texture2D(u_deltaNTex, tc).r;

        if (u_colormap == 4 || u_colormap == 5) {
          // Index black (0) -> green (max), with the light composited on top
          vec3 indexGreen = mix(vec3(0.0), vec3(0.1, 0.8, 0.3), deltaN);
          // Muted like the Electric Cyan/Gold index tint (same smoothstep and 0.32 weight), so
          // the light stands out; the index view during training shows the full green
          vec3 indexMuted = vec3(0.1, 0.8, 0.3) * smoothstep(0.05, 0.70, deltaN) * 0.32;
          vec3 lit;
          if (u_colormap == 4) {
            // Red light, opacity = intensity (transparent at 0, opaque red at the maximum)
            lit = mix(indexMuted, vec3(1.0, 0.05, 0.05), intensity);
          } else {
            // Thermal-camera ("ironbow") light: hue and brightness both rise with intensity;
            // opacity = sqrt(intensity) so weak light stays visible but 0 is transparent
            lit = mix(indexMuted, colormapIronbow(intensity), clamp(1.3 * sqrt(intensity), 0.0, 1.0));
          }
          // Index view (training): fade the light out so the green index map shows
          gl_FragColor = vec4(mix(lit, indexGreen, u_indexView), 1.0);
          return;
        }

        // Apply optical colormap
        vec3 color = vec3(0.0);
        if (u_colormap == 0) {
          color = colormapInferno(intensity);
        } else if (u_colormap == 1) {
          color = colormapViridis(intensity);
        } else if (u_colormap == 2) {
          color = colormapCyanGold(intensity);
        } else {
          color = colormapGreenLaser(intensity);
        }

        // Overlay refractive index perturbation as a sleek, non-saturating waveguide core
        float n_factor = smoothstep(0.05, 0.70, deltaN);
        vec3 indexTint = vec3(0.08, 0.38, 0.70) * n_factor * 0.32;
        color = clamp(color + indexTint * (1.0 - color * 0.75), 0.0, 1.0);

        vec3 indexColor = color * 0.3 + vec3(0.25, 0.6, 1.0) * pow(deltaN, 0.8);
        color = mix(color, clamp(indexColor, 0.0, 1.0), u_indexView);

        gl_FragColor = vec4(color, 1.0);
      }
    `;

    const vertShader = this._compileShader(gl.VERTEX_SHADER, vsSource);
    const fragShader = this._compileShader(gl.FRAGMENT_SHADER, fsSource);

    this.program = gl.createProgram();
    gl.attachShader(this.program, vertShader);
    gl.attachShader(this.program, fragShader);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Shader link error:', gl.getProgramInfoLog(this.program));
      this.useWebGL = false;
      this.ctx2d = this.canvas.getContext('2d');
      return;
    }

    gl.useProgram(this.program);

    this.uIntensityTexLoc = gl.getUniformLocation(this.program, 'u_intensityTex');
    this.uDeltaNTexLoc = gl.getUniformLocation(this.program, 'u_deltaNTex');
    this.uColormapLoc = gl.getUniformLocation(this.program, 'u_colormap');
    this.uPulsedLoc = gl.getUniformLocation(this.program, 'u_pulsed');
    this.uTimeLoc = gl.getUniformLocation(this.program, 'u_time');
    this.uTransitLoc = gl.getUniformLocation(this.program, 'u_transit');
    this.uPeriodLoc = gl.getUniformLocation(this.program, 'u_period');
    this.uFwhmLoc = gl.getUniformLocation(this.program, 'u_fwhm');
    this.uIndexViewLoc = gl.getUniformLocation(this.program, 'u_indexView');
    this.uLayersLoc = gl.getUniformLocation(this.program, 'u_layers');
    this.uRepeatLoc = gl.getUniformLocation(this.program, 'u_repeat');
    this.uSwitchKLoc = gl.getUniformLocation(this.program, 'u_switchK');
    this.uOldRuleLoc = gl.getUniformLocation(this.program, 'u_oldRule');
    this.uNewRuleLoc = gl.getUniformLocation(this.program, 'u_newRule');
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  _initBuffers() {
    if (!this.useWebGL) return;
    const gl = this.gl;

    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const posAttr = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(posAttr);
    gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);
  }

  _initTextures() {
    if (!this.useWebGL) return;
    const gl = this.gl;

    // Unit 0: Active Display Intensity
    this.intensityTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.intensityTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Unit 1: Delta N
    this.deltaNTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.deltaNTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  updateGridSize(nx, nz, layers = 1) {
    if (this.nx === nx && this.nz === nz && this.layers === layers) return;
    this.nx = nx;
    this.nz = nz;
    this.layers = layers;
    this.displayIntensity = new Uint8Array(nx * nz * layers);
    this.latestIntensity = null; // next frame is shown in full instead of swept in
    this.morph = null;
    this.columnVersions = new Uint32Array(nz);
    this.activeWavefronts = [];
    this.speedColsPerMs = nz / 1400.0;
    if (this.offscreenCanvas) {
      this.offscreenCanvas = null;
    }
  }

  setColormap(colormapName) {
    const map = {
      inferno: 0,
      viridis: 1,
      cyan_gold: 2,
      green_laser: 3,
      green_index_red_light: 4,
      green_index_thermal_light: 5,
    };
    this.colormapId = map[colormapName] !== undefined ? map[colormapName] : 0;
  }

  setPulsed(pulsed) {
    this.pulsed = pulsed;
  }

  setIndexView(on) {
    this.indexViewTarget = on ? 1.0 : 0.0;
  }

  /** Pulse-train time in seconds since page load (float32 keeps ms precision for many hours). */
  _pulseTime() {
    return (performance.now() - this.timeOrigin) / 1000.0;
  }

  _layerOfPulse(k) {
    const rule = k >= this.switchPulse ? this.newRule : this.oldRule;
    if (rule >= 0) return Math.min(rule, this.layers - 1);
    const n = Math.floor(k / this.pulsesPerInput);
    return ((n % this.layers) + this.layers) % this.layers;
  }

  /** Input layer carried by the pulse nearest to fractional position zFrac (0 = input facet). */
  pulseLayerAt(zFrac) {
    const tau = this._pulseTime() - zFrac * this.pulseTransitSec;
    return this._layerOfPulse(Math.floor(tau / this.pulsePeriodSec + 0.5));
  }

  /**
   * Manual mode: send `layer` with every pulse from the next one launched on; -1 returns to
   * cycling through the inputs. `immediate` (demo switch) applies it to all pulses at once.
   */
  setManualLayer(layer, immediate = false) {
    if (immediate) {
      this.oldRule = layer;
      this.newRule = layer;
      this.switchPulse = -1e9;
      return;
    }
    // The pulse nearest the input facet has already started entering; switch after it
    const kNow = Math.floor(this._pulseTime() / this.pulsePeriodSec + 0.5);
    this.oldRule = this.newRule;
    this.newRule = layer;
    this.switchPulse = kNow + 1;
  }

  /** Intensity envelope of the pulse train at fractional position zFrac in [0, 1]. */
  _pulseEnvelope(zFrac, t) {
    const P = this.pulsePeriodSec;
    const tau = t - zFrac * this.pulseTransitSec;
    let d = tau / P + 0.5;
    d = (d - Math.floor(d) - 0.5) * P;
    return Math.exp(-2.7725887 * d * d / (this.pulseFwhmSec * this.pulseFwhmSec));
  }

  /**
   * Enqueue a new progressive wavefront from an optical simulation update.
   * Multiple wavefronts are buffered and advance simultaneously without abrupt jumping.
   */
  /** Re-express the on-screen intensity in units of a new frame's global max. */
  _rescaleDisplay(newMax) {
    if (!newMax) return;
    if (this.displayMax && this.displayMax !== newMax) {
      const f = this.displayMax / newMax;
      const d = this.displayIntensity;
      for (let i = 0; i < d.length; i++) d[i] = Math.min(255, Math.round(d[i] * f));
    }
    this.displayMax = newMax;
  }

  enqueueWavefront(uint8Array, startZFrac = 0.0, meta = null, maxVal = null) {
    this.currentUpdateId++;
    const updateId = this.currentUpdateId;
    const startCol = Math.max(0, Math.min(this.nz - 1, Math.floor(startZFrac * this.nz)));

    // On the initial frame load, immediately display the complete pattern
    if (!this.latestIntensity) {
      this.displayMax = maxVal;
      this.displayIntensity.set(uint8Array);
      this.columnVersions.fill(updateId);
      this.latestIntensity = new Uint8Array(uint8Array);
      this._uploadDisplayTexture();
      if (meta && this.onWavefrontReachEnd) {
        this.onWavefrontReachEnd(meta);
      }
      return;
    }

    this.latestIntensity = new Uint8Array(uint8Array);
    this._rescaleDisplay(maxVal);

    // Enqueue new wavefront starting at startCol
    this.activeWavefronts.push({
      id: updateId,
      targetIntensity: new Uint8Array(uint8Array),
      max: maxVal,
      startCol: startCol,
      currentCol: startCol,
      writtenStart: false,
      meta: meta,
      reachedEnd: false,
    });

    this.textureNeedsUpload = true;
  }

  /**
   * Backward-compatible alias for enqueueWavefront.
   */
  uploadIntensityBuffer(uint8Array, triggerWavefront = false, startZFrac = 0.0, meta = null) {
    this.enqueueWavefront(uint8Array, startZFrac, meta);
  }

  /**
   * Cross-fade intensity and index map to new frames over durationMs (used while
   * training streams one frame per optimizer step). Cancels pending wavefronts.
   */
  morphTo(intensity, deltaN, durationMs, maxVal = null) {
    const n = this.nx * this.nz;
    this._rescaleDisplay(maxVal);
    // Start from what is on screen, including a half-finished previous morph
    let shownDn = this.latestDeltaN;
    if (this.morph) shownDn = this.morph.elapsed > 0 ? this.morph.blendDn : this.morph.fromDn;
    this.activeWavefronts = [];
    this.currentUpdateId++;
    this.columnVersions.fill(this.currentUpdateId);
    this.latestIntensity = new Uint8Array(intensity);
    this.morph = {
      fromI: new Uint8Array(this.displayIntensity),
      toI: this.latestIntensity,
      fromDn: shownDn && shownDn.length === n ? new Uint8Array(shownDn) : new Uint8Array(deltaN),
      toDn: new Uint8Array(deltaN),
      blendDn: new Uint8Array(n),
      elapsed: 0,
      duration: durationMs,
    };
  }

  _advanceMorph(dtMs) {
    const m = this.morph;
    m.elapsed += dtMs;
    const a = Math.min(1.0, m.elapsed / m.duration);
    for (let i = 0; i < m.toI.length; i++) {
      this.displayIntensity[i] = m.fromI[i] + (m.toI[i] - m.fromI[i]) * a;
    }
    for (let i = 0; i < m.toDn.length; i++) {
      m.blendDn[i] = m.fromDn[i] + (m.toDn[i] - m.fromDn[i]) * a;
    }
    this._uploadDeltaNTexture(m.blendDn);
    this._uploadDisplayTexture();
    if (a >= 1.0) {
      this.latestDeltaN = m.toDn;
      this.morph = null;
    }
  }

  uploadDeltaNBuffer(uint8Array) {
    this.morph = null;
    this.latestDeltaN = uint8Array;
    this._uploadDeltaNTexture(uint8Array);
  }

  _uploadDeltaNTexture(uint8Array) {
    if (!this.useWebGL) return;

    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.deltaNTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const format = this.isWebGL2 ? gl.RED : gl.LUMINANCE;
    const internal = this.isWebGL2 ? gl.R8 : gl.LUMINANCE;
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, this.nz, this.nx, 0, format, gl.UNSIGNED_BYTE, uint8Array);
  }

  /**
   * Advance all active wavefronts across the display buffer.
   * Invoked each frame in the render loop.
   */
  update(dtMs) {
    // Ease the index view in/out (~0.5 s time constant)
    this.indexView += (this.indexViewTarget - this.indexView) * Math.min(1.0, dtMs / 500.0);

    if (this.morph) {
      this._advanceMorph(dtMs);
      return true;
    }

    if (this.activeWavefronts.length === 0) {
      if (this.textureNeedsUpload) {
        this._uploadDisplayTexture();
        this.textureNeedsUpload = false;
        return true;
      }
      return false;
    }

    // Safeguard: keep active wavefront queue bounded to prevent memory build-up during rapid edits
    if (this.activeWavefronts.length > 12) {
      const pruned = [this.activeWavefronts[0]];
      for (let i = 1; i < this.activeWavefronts.length - 1; i++) {
        const curr = this.activeWavefronts[i];
        const next = this.activeWavefronts[i + 1];
        if (Math.abs(curr.currentCol - next.currentCol) >= 4.0) {
          pruned.push(curr);
        }
      }
      pruned.push(this.activeWavefronts[this.activeWavefronts.length - 1]);
      this.activeWavefronts = pruned;
    }

    const step = dtMs * this.speedColsPerMs;
    let hasChanged = false;

    for (let i = 0; i < this.activeWavefronts.length; i++) {
      const wf = this.activeWavefronts[i];
      const prevCol = Math.floor(wf.currentCol);
      wf.currentCol = Math.min(this.nz, wf.currentCol + step);
      const nextCol = Math.min(this.nz - 1, Math.floor(wf.currentCol));

      const fromCol = wf.writtenStart ? prevCol + 1 : wf.startCol;
      wf.writtenStart = true;

      if (nextCol >= fromCol) {
        for (let c = fromCol; c <= nextCol; c++) {
          // Physical Invariant: only write if this wavefront is newer than or equal to column version
          if (wf.id >= this.columnVersions[c]) {
            this.columnVersions[c] = wf.id;
            // An older wavefront still in flight may use a different max than the display
            const f = wf.max && this.displayMax ? wf.max / this.displayMax : 1.0;
            for (let r = 0; r < this.nx * this.layers; r++) {
              const idx = r * this.nz + c;
              const v = wf.targetIntensity[idx];
              this.displayIntensity[idx] = f === 1.0 ? v : Math.min(255, Math.round(v * f));
            }
            hasChanged = true;
          }
        }
      }

      // Check if this wavefront reached the output facet
      if (wf.currentCol >= this.nz - 1 && !wf.reachedEnd) {
        wf.reachedEnd = true;
        if (wf.meta && this.onWavefrontReachEnd) {
          this.onWavefrontReachEnd(wf.meta);
        }
      }
    }

    // Clean up wavefronts that have fully traversed the waveguide
    this.activeWavefronts = this.activeWavefronts.filter((wf) => wf.currentCol < this.nz);

    if (hasChanged || this.textureNeedsUpload) {
      this._uploadDisplayTexture();
      this.textureNeedsUpload = false;
      return true;
    }

    return false;
  }

  _uploadDisplayTexture() {
    if (!this.useWebGL) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.intensityTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const format = this.isWebGL2 ? gl.RED : gl.LUMINANCE;
    const internal = this.isWebGL2 ? gl.R8 : gl.LUMINANCE;
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, this.nz, this.nx * this.layers, 0, format, gl.UNSIGNED_BYTE, this.displayIntensity);
  }

  render() {
    if (this.canvas.width !== this.canvas.clientWidth || this.canvas.height !== this.canvas.clientHeight) {
      this.canvas.width = this.canvas.clientWidth;
      this.canvas.height = this.canvas.clientHeight;
    }

    if (this.useWebGL && this.gl) {
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);

      gl.useProgram(this.program);
      gl.uniform1i(this.uIntensityTexLoc, 0);
      gl.uniform1i(this.uDeltaNTexLoc, 1);
      gl.uniform1i(this.uColormapLoc, this.colormapId);
      gl.uniform1f(this.uPulsedLoc, this.pulsed ? 1.0 : 0.0);
      gl.uniform1f(this.uTimeLoc, this._pulseTime());
      gl.uniform1f(this.uTransitLoc, this.pulseTransitSec);
      gl.uniform1f(this.uPeriodLoc, this.pulsePeriodSec);
      gl.uniform1f(this.uFwhmLoc, this.pulseFwhmSec);
      gl.uniform1f(this.uIndexViewLoc, this.indexView);
      gl.uniform1f(this.uLayersLoc, this.layers);
      gl.uniform1f(this.uRepeatLoc, this.pulsesPerInput);
      gl.uniform1f(this.uSwitchKLoc, this.switchPulse);
      gl.uniform1f(this.uOldRuleLoc, this.oldRule);
      gl.uniform1f(this.uNewRuleLoc, this.newRule);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    } else if (this.ctx2d && this.displayIntensity) {
      this._renderCanvas2D();
    }
  }

  _renderCanvas2D() {
    const ctx = this.ctx2d;
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (!this.offscreenCanvas) {
      this.offscreenCanvas = document.createElement('canvas');
      this.offscreenCanvas.width = this.nz;
      this.offscreenCanvas.height = this.nx;
      this.offscreenCtx = this.offscreenCanvas.getContext('2d');
      this.imgData = this.offscreenCtx.createImageData(this.nz, this.nx);
    }

    const img = this.imgData;
    const intensity = this.displayIntensity;
    const deltaN = this.latestDeltaN;

    if (!this.colEnvelope || this.colEnvelope.length !== this.nz) {
      this.colEnvelope = new Float32Array(this.nz);
    }
    const t = this._pulseTime();
    for (let col = 0; col < this.nz; col++) {
      this.colEnvelope[col] = this.pulsed ? this._pulseEnvelope((col + 0.5) / this.nz, t) : 1.0;
    }

    for (let row = 0; row < this.nx; row++) {
      const flipRow = row; // texture row 0 (x = -Lx/2) at the top, as in the shader
      for (let col = 0; col < this.nz; col++) {
        const srcIdx = flipRow * this.nz + col;
        const dstIdx = (row * this.nz + col) * 4;

        let val = (intensity ? intensity[srcIdx] : 0) / 255.0 * this.colEnvelope[col];
        const dn = deltaN ? (deltaN[srcIdx] / 255.0) : 0.0;

        let r = Math.min(255, val * 320 + dn * 40);
        let g = Math.min(255, val > 0.4 ? (val - 0.4) * 400 + dn * 150 : dn * 80);
        let b = Math.min(255, val > 0.7 ? (val - 0.7) * 800 + dn * 255 : dn * 200);

        img.data[dstIdx] = r;
        img.data[dstIdx + 1] = g;
        img.data[dstIdx + 2] = b;
        img.data[dstIdx + 3] = 255;
      }
    }
    this.offscreenCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.offscreenCanvas, 0, 0, w, h);
  }
}

