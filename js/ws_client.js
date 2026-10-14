/**
 * High-Speed Binary WebSocket Client.
 * Streams JSON state and Float32Array buffers with sub-15ms roundtrip latency.
 */

export class WaveguideWSClient {
  constructor(options = {}) {
    this.onState = options.onState || (() => {});
    this.onIntensityBuffer = options.onIntensityBuffer || (() => {});
    this.onDeltaNBuffer = options.onDeltaNBuffer || (() => {});
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onTrainResult = options.onTrainResult || (() => {});

    this.ws = null;
    this.isConnected = false;

    this.connect();
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${protocol}//${host}/ws`;

    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.isConnected = true;
      this.onStatusChange(true, 'Connected to PyTorch BPM Engine');
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      this.onStatusChange(false, 'Disconnected (Reconnecting...)');
      setTimeout(() => this.connect(), 2000);
    };

    this.ws.onerror = (err) => {
      console.warn('WebSocket error:', err);
      this.ws.close();
    };

    this.ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'state') {
            this.onState(msg.data);
          } else if (msg.type === 'train_result') {
            this.onTrainResult(msg.metrics);
          }
        } catch (e) {
          console.error('Failed to parse JSON message:', e);
        }
      } else if (event.data instanceof ArrayBuffer) {
        const buffer = event.data;
        if (buffer.byteLength >= 8) {
          const headerBytes = new Uint8Array(buffer, 0, 8);
          const headerStr = String.fromCharCode(...headerBytes);

          if (headerStr === 'INTENS00') {
            const uint8Array = new Uint8Array(buffer, 8);
            this.onIntensityBuffer(uint8Array);
          } else if (headerStr === 'DELTAN00') {
            const uint8Array = new Uint8Array(buffer, 8);
            this.onDeltaNBuffer(uint8Array);
          }
        }
      }
    };
  }

  /** Switch the demo (y_splitter, mzi, mnist, perm4, custom, ...). */
  sendDemo(name) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'demo', name }));
    }
  }

  sendRouting(n, connections) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'routing', n, connections }));
    }
  }

  sendTrain() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'train' }));
    }
  }
}
