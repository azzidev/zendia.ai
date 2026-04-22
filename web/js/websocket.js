const WS = {
  socket: null,
  state: {
    epoch: 0, step: 0, loss: 1, avgLoss: 1,
    tokensSeen: 0, architecture: [], sample: '', phase: '',
    layers: []
  },
  lossHistory: [],
  listeners: [],

  connect(host) {
    const url = 'ws://' + (host || location.host) + '/ws';
    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      document.getElementById('status-dot').classList.add('connected');
      document.getElementById('status-text').textContent = 'conectado';
    };

    this.socket.onclose = () => {
      document.getElementById('status-dot').classList.remove('connected');
      document.getElementById('status-text').textContent = 'desconectado';
      // reconecta em 3s
      setTimeout(() => this.connect(host), 3000);
    };

    this.socket.onmessage = (e) => {
      const data = JSON.parse(e.data);
      Object.assign(this.state, data);

      if (data.loss !== undefined) {
        this.lossHistory.push(data.loss);
        if (this.lossHistory.length > 1000) {
          this.lossHistory = this.lossHistory.slice(-1000);
        }
      }

      this.listeners.forEach(fn => fn(this.state, data));
    };
  },

  onUpdate(fn) {
    this.listeners.push(fn);
  }
};
