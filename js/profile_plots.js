/**
 * High-performance 1D Beam Profile Plotter.
 * Renders transverse beam intensity I(x) vertically aligned with the 2D waveguide facets.
 */

export class ProfilePlotter {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.color = options.color || '#00f2fe';
    this.fillColor = options.fillColor || 'rgba(0, 242, 254, 0.2)';
    this.label = options.label || 'Intensity';
    this.isInput = options.isInput || false;
    this.title = options.title || '';

    this.profileData = null;
    this.maxVal = 1.0;
    this.lx = 600.0;
    this.bins = null; // {edges: [11 x positions in um], powers: [10], winner}
    this.pixels = null; // {values: [49 in 0..1], xMin, xMax} drawn at the outer edge (input)
    this.target = null; // optional target profile, drawn dashed on the same scale
  }

  setTarget(profile) {
    this.target = profile;
    this.render();
  }

  /** Show the unrolled MNIST pixels, each next to the x of the input mode it drives. */
  setPixels(pixels) {
    this.pixels = pixels;
    this.render();
  }

  _drawPixels(ctx, h, x0, width) {
    const { values, xMin, xMax } = this.pixels;
    const n = values.length;
    const pitch = ((xMax - xMin) / (n - 1) / this.lx) * h;
    const yOf = (x) => h * (0.5 + x / this.lx);
    ctx.save();
    for (let k = 0; k < n; k++) {
      const yc = yOf(xMin + ((xMax - xMin) * k) / (n - 1));
      const v = Math.round(Math.min(1, Math.max(0, values[k])) * 255);
      ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
      ctx.fillRect(x0, yc - pitch / 2, width, pitch);
    }
    // Mark the start of each 7-pixel image row
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)';
    ctx.lineWidth = 1;
    const side = Math.round(Math.sqrt(n));
    for (let r = 0; r <= side; r++) {
      const y = yOf(xMin + ((xMax - xMin) * (r * side - 0.5)) / (n - 1));
      ctx.beginPath();
      ctx.moveTo(x0 - 1, y);
      ctx.lineTo(x0 + width + 1, y);
      ctx.stroke();
    }
    ctx.strokeRect(x0 - 0.5, yOf(xMin) - pitch / 2, width + 1, yOf(xMax) - yOf(xMin) + pitch);
    ctx.restore();
  }

  /** Show detector bins (edges in um, powers normalized) behind the profile, or null to hide. */
  setBins(bins) {
    this.bins = bins;
    this.render();
  }

  setLx(lx) {
    this.lx = lx;
  }

  _drawBins(ctx, w, h) {
    const { edges, powers, winner } = this.bins;
    const labelOffset = this.bins.labelOffset || 0; // demos number outputs from 1
    const yOf = (x) => h * (0.5 + x / this.lx); // x axis points down
    ctx.save();
    ctx.font = '600 10px JetBrains Mono, monospace';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < edges.length - 1; i++) {
      const yTop = yOf(edges[i]);
      const yBot = yOf(edges[i + 1]);
      const isWin = i === winner;
      ctx.fillStyle = isWin ? 'rgba(250, 204, 21, 0.28)' : (i % 2 ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.025)');
      ctx.fillRect(0, yTop, w, yBot - yTop);
      ctx.strokeStyle = isWin ? 'rgba(250, 204, 21, 0.9)' : 'rgba(255, 255, 255, 0.12)';
      ctx.lineWidth = isWin ? 1.5 : 1;
      ctx.strokeRect(0.5, yTop + 0.5, w - 1, yBot - yTop - 1);
      // Digit label and detected power share at the outer edge
      ctx.fillStyle = isWin ? '#facc15' : '#64748b';
      ctx.textAlign = this.isInput ? 'right' : 'left';
      const lx = this.isInput ? w - 4 : 4;
      const pct = powers ? ` ${(powers[i] * 100).toFixed(0)}%` : '';
      ctx.fillText(`${i + labelOffset}${pct}`, lx, (yTop + yBot) / 2);
    }
    ctx.restore();
  }

  setData(profileArray, maxVal = null) {
    this.profileData = profileArray;
    if (maxVal) {
      this.maxVal = maxVal;
    } else if (profileArray && profileArray.length > 0) {
      this.maxVal = Math.max(...profileArray, 1e-6);
    }
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Clear background
    ctx.fillStyle = '#060a12';
    ctx.fillRect(0, 0, w, h);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;

    // Horizontal center line (x = 0)
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Secondary grid lines
    ctx.beginPath();
    ctx.moveTo(0, h * 0.25);
    ctx.lineTo(w, h * 0.25);
    ctx.moveTo(0, h * 0.75);
    ctx.lineTo(w, h * 0.75);
    ctx.stroke();

    if (this.bins) this._drawBins(ctx, w, h);

    // With pixels shown, the outer edge holds the pixel column and everything shifts inward
    const inset = this.pixels ? 18 : 0;
    if (this.pixels) this._drawPixels(ctx, h, 3, 13);

    if (this.title) {
      // Title runs vertically along the outer edge of the strip
      ctx.save();
      ctx.fillStyle = '#94a3b8';
      ctx.font = '600 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.translate(this.isInput ? 9 + inset : w - 9, h / 2);
      ctx.rotate(this.isInput ? -Math.PI / 2 : Math.PI / 2);
      ctx.fillText(this.pixels ? 'PIXELS ROW BY ROW  →  INPUT I(x)' : this.title, 0, 3);
      ctx.restore();
    }

    if (!this.profileData || this.profileData.length === 0) return;

    const n = this.profileData.length;
    // Baseline sits on the side away from the 2D display; the profile grows toward it
    const paddingLeft = this.isInput ? 18 + inset : 6;
    const paddingRight = this.isInput ? 6 : 18;
    const plotWidth = w - paddingLeft - paddingRight;

    ctx.save();

    // Build path: y goes from h (x = -Lx/2) to 0 (x = +Lx/2)
    // intensity extends horizontally from left to right (or right to left if right panel)
    ctx.beginPath();

    const startX = this.isInput ? paddingLeft : w - paddingRight;

    for (let i = 0; i < n; i++) {
      // Index 0 is x = -Lx/2 (top of screen), index n-1 is +Lx/2 (bottom); x axis points down
      const y = (i / (n - 1)) * h;
      const normVal = Math.min(1.0, this.profileData[i] / (this.maxVal || 1.0));
      const valOffset = normVal * plotWidth * 0.85;

      const x = this.isInput ? startX + valOffset : startX - valOffset;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    // Stroke the glow curve
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 2.2;
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 8;
    ctx.stroke();

    // Fill under curve
    ctx.shadowBlur = 0;
    if (this.isInput) {
      ctx.lineTo(paddingLeft, 0);
      ctx.lineTo(paddingLeft, h);
    } else {
      ctx.lineTo(w - paddingRight, 0);
      ctx.lineTo(w - paddingRight, h);
    }
    ctx.closePath();
    ctx.fillStyle = this.fillColor;
    ctx.fill();

    if (this.target && this.target.length === n) {
      // Target output (dashed), same scale as the simulated profile
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const y = (i / (n - 1)) * h;
        const off = Math.min(1.0, this.target[i] / (this.maxVal || 1.0)) * plotWidth * 0.85;
        const x = this.isInput ? startX + off : startX - off;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw coordinate labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = this.isInput ? 'left' : 'right';
    const labelX = this.isInput ? paddingLeft + 4 : w - paddingRight - 4;
    ctx.fillText(`-${this.lx / 2} μm`, labelX, 14);
    ctx.fillText('0 μm', labelX, h / 2 - 4);
    ctx.fillText(`+${this.lx / 2} μm`, labelX, h - 6);

    ctx.restore();
  }

  resize() {
    if (this.canvas.width !== this.canvas.clientWidth || this.canvas.height !== this.canvas.clientHeight) {
      this.canvas.width = this.canvas.clientWidth;
      this.canvas.height = this.canvas.clientHeight;
      this.render();
    }
  }
}
