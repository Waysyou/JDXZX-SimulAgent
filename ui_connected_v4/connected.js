(() => {
  "use strict";

  // 直接双击 index.html 时保留原来的纯前端样板行为；只有通过本地服务打开才接管真实功能。
  if (!/^https?:$/.test(window.location.protocol)) return;

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    ignition: $("#ignition"),
    ignitionLabel: $("#ignitionLabel"),
    ignitionNote: $("#ignitionNote"),
    heroStatus: $("#heroStatus"),
    heroDot: $("#heroDot"),
    controlCard: $("#controlCard"),
    latency: $("#latency"),
    stableCount: $("#stableCount"),
    queueCount: $("#queueCount"),
    inputState: $("#inputState"),
    inputDetail: $("#inputDetail"),
    semanticTable: $("#semanticTable"),
    liveDataTag: $("#liveDataTag"),
    playButton: $("#playButton"),
    audioBadge: $("#audioBadge"),
    nowPlaying: $("#nowPlaying"),
    trackFill: $("#trackFill"),
    duration: $("#duration"),
    selectWavButton: $("#selectWavButton"),
    wavFileInput: $("#wavFileInput"),
    sourceAudio: $("#sourceAudio"),
    ttsAudio: $("#ttsAudio"),
    toast: $("#toast"),
    uploadPanel: $("#uploadPanel"),
    backendFileList: $("#backendFileList"),
    systemModuleGrid: $("#systemModuleGrid"),
    waveform: $("#liveWaveform"),
  };

  const runtime = {
    phase: "idle",
    mode: "mic",
    modelReady: false,
    sessionId: null,
    selectedFile: null,
    sourceObjectUrl: null,
    stream: null,
    audioContext: null,
    mediaSource: null,
    processor: null,
    silentGain: null,
    visualAnalyser: null,
    visualData: null,
    visualFrame: null,
    fileAudioContext: null,
    fileMediaSource: null,
    fileAnalyser: null,
    pendingSamples: [],
    pendingSampleCount: 0,
    sendChain: Promise.resolve(),
    pollTimer: null,
    audioQueue: [],
    queuedSegmentIds: new Set(),
    playingSegmentId: null,
    currentSnapshot: null,
    toastTimer: null,
  };

  const tr = (key, fallback = "") => window.V4Settings?.t(key, fallback) || fallback;

  const semanticHeader = () => `
    <div class="semantic-row header">
      <div class="semantic-cell">${tr("semantic.time", "时间")}</div><div class="semantic-cell">${tr("semantic.chinese", "中文 · 已确认")}</div><div class="semantic-cell">${tr("semantic.english", "English · 点击本行播放")}</div><div class="semantic-cell">${tr("semantic.segmentId", "片段 ID")}</div>
    </div>`;

  function showToast(message, timeout = 2200) {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    window.clearTimeout(runtime.toastTimer);
    runtime.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), timeout);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) payload = await response.json();
    if (!response.ok) {
      const message = payload?.detail || payload?.message || `请求失败（${response.status}）`;
      throw new Error(message);
    }
    return payload;
  }

  function selectedMode() {
    return $(".segmented button.active")?.dataset.mode === "file" ? "file" : "mic";
  }

  function resetWaveform() {
    if (runtime.visualFrame) window.cancelAnimationFrame(runtime.visualFrame);
    runtime.visualFrame = null;
    runtime.visualAnalyser = null;
    runtime.visualData = null;
    elements.waveform.classList.remove("monitoring");
    elements.waveform.querySelectorAll("span").forEach((bar) => {
      bar.style.height = "4px";
    });
  }

  function startWaveform(analyser) {
    if (runtime.visualFrame) window.cancelAnimationFrame(runtime.visualFrame);
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.72;
    runtime.visualAnalyser = analyser;
    runtime.visualData = new Uint8Array(analyser.fftSize);
    elements.waveform.classList.add("monitoring");
    const bars = [...elements.waveform.querySelectorAll("span")];

    const draw = () => {
      if (runtime.visualAnalyser !== analyser) return;
      analyser.getByteTimeDomainData(runtime.visualData);
      const stride = Math.max(1, Math.floor(runtime.visualData.length / bars.length));
      bars.forEach((bar, index) => {
        const start = index * stride;
        const end = Math.min(runtime.visualData.length, start + stride);
        let peak = 0;
        for (let sample = start; sample < end; sample += 1) {
          peak = Math.max(peak, Math.abs(runtime.visualData[sample] - 128) / 128);
        }
        const height = 4 + Math.min(1, peak * 3.4) * 46;
        bar.style.height = `${height.toFixed(1)}px`;
      });
      runtime.visualFrame = window.requestAnimationFrame(draw);
    };
    draw();
  }

  async function prepareFileWaveform() {
    if (!runtime.fileAudioContext || runtime.fileAudioContext.state === "closed") {
      runtime.fileAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      runtime.fileMediaSource = runtime.fileAudioContext.createMediaElementSource(elements.sourceAudio);
      runtime.fileAnalyser = runtime.fileAudioContext.createAnalyser();
      runtime.fileMediaSource.connect(runtime.fileAnalyser);
      runtime.fileAnalyser.connect(runtime.fileAudioContext.destination);
    }
    await runtime.fileAudioContext.resume();
    startWaveform(runtime.fileAnalyser);
  }

  function setPhase(phase, note) {
    runtime.phase = phase;
    elements.ignition.disabled = phase === "loading" || phase === "stopping";
    elements.ignition.className = phase === "ready" ? "ignition ready" : phase === "live" ? "ignition live" : "ignition";
    elements.controlCard.classList.toggle("live", phase === "live");

    const copy = [
      tr(`phase.${phase}.button`),
      note || tr(`phase.${phase}.note`),
      tr(`phase.${phase}.status`),
      tr(`phase.${phase}.state`),
    ];
    elements.ignitionLabel.textContent = copy[0];
    elements.ignitionNote.textContent = copy[1];
    elements.heroStatus.textContent = copy[2];
    elements.inputState.textContent = copy[3];
    elements.heroDot.style.background = phase === "live" ? "var(--success)" : phase === "ready" ? "var(--blue)" : "";
  }

  function formatTime(value) {
    const raw = String(value || "").replace(/[\[\]]/g, "").trim();
    return raw || "--:--.--";
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function renderRows(rows) {
    const body = rows.length ? rows.map((row) => {
      const audioClass = row.has_audio ? " has-audio" : "";
      return `<div class="semantic-row data${audioClass}" tabindex="0" data-segment-id="${Number(row.segment_id)}">
        <div class="semantic-cell">${escapeHtml(formatTime(row.time))}</div>
        <div class="semantic-cell">${escapeHtml(row.chinese)}</div>
        <div class="semantic-cell">${escapeHtml(row.english)}</div>
        <div class="semantic-cell">S-${String(row.segment_id).padStart(3, "0")}</div>
      </div>`;
    }).join("") : `<div class="semantic-row empty"><div class="semantic-cell">${tr("semantic.empty", "启动同传后，稳定的中英文片段会显示在这里")}</div></div>`;
    elements.semanticTable.innerHTML = semanticHeader() + body;
  }

  function clearLiveOutput() {
    elements.ttsAudio.pause();
    elements.ttsAudio.removeAttribute("src");
    elements.ttsAudio.load();
    elements.latency.textContent = "—";
    elements.stableCount.textContent = "0";
    elements.queueCount.textContent = "0 / 4";
    elements.liveDataTag.innerHTML = `<i></i>${tr("semantic.waiting", "等待输入")}`;
    elements.audioBadge.textContent = tr("audio.waiting", "等待播放");
    elements.playButton.textContent = "▶";
    elements.trackFill.style.width = "0%";
    elements.duration.textContent = "00:00 / 00:00";
    elements.nowPlaying.textContent = tr("audio.none", "尚无英文语音");
    renderRows([]);
    renderQueue({ rows: [], played_segment_ids: [] });
  }

  function renderQueue(snapshot) {
    const list = $(".queue-list");
    if (!list) return;
    const rows = snapshot.rows.filter((row) => row.has_audio).slice(-4);
    if (!rows.length) {
      list.innerHTML = `<div class="queue-item"><span class="queue-index">—</span><span>${tr("queue.waiting", "等待英文语音")}</span><span class="queue-state">${tr("queue.idle", "空闲")}</span></div>`;
      return;
    }
    const played = new Set((snapshot.played_segment_ids || []).map(Number));
    list.innerHTML = rows.map((row, index) => {
      const id = Number(row.segment_id);
      const active = id === runtime.playingSegmentId;
      const state = active ? "正在播放" : played.has(id) ? "已播放" : "等待";
      return `<div class="queue-item${active ? " active" : ""}"><span class="queue-index">${String(index + 1).padStart(2, "0")}</span><span>S-${String(id).padStart(3, "0")}</span><span class="queue-state">${state}</span></div>`;
    }).join("");
  }

  function applySnapshot(snapshot) {
    runtime.currentSnapshot = snapshot;
    runtime.sessionId = snapshot.session_id;
    elements.latency.textContent = snapshot.latency > 0 ? `${snapshot.latency.toFixed(2)} s` : "—";
    elements.stableCount.textContent = String(snapshot.stable_count || 0);
    elements.queueCount.textContent = `${snapshot.queue_size || 0} / ${snapshot.queue_max || 4}`;
    elements.inputState.textContent = snapshot.input_state || (runtime.phase === "live" ? "监听中" : "就绪");
    elements.inputDetail.textContent = `${Number(snapshot.input_seconds || 0).toFixed(1)} 秒音频`;
    elements.heroStatus.textContent = snapshot.status || elements.heroStatus.textContent;
    elements.liveDataTag.innerHTML = "<i></i>实时数据";
    renderRows(snapshot.rows || []);
    renderQueue(snapshot);
    enqueueAudioEvents(snapshot.audio_events || []);
    if (snapshot.last_error) showToast(snapshot.last_error, 4200);
  }

  function highlightSegment(segmentId) {
    document.querySelectorAll("#semanticTable .semantic-row.data").forEach((row) => {
      row.classList.toggle("selected", Number(row.dataset.segmentId) === Number(segmentId));
    });
  }

  function enqueueAudioEvents(events) {
    for (const event of events) {
      const id = Number(event.segment_id);
      if (!id || runtime.queuedSegmentIds.has(id)) continue;
      runtime.queuedSegmentIds.add(id);
      runtime.audioQueue.push({ id, url: event.url });
    }
    const autoplay = window.V4Settings?.get().autoplay ?? true;
    if (autoplay && runtime.audioQueue.length && elements.ttsAudio.paused && !runtime.playingSegmentId) playNextAudio();
  }

  async function playAudioItem(item) {
    runtime.playingSegmentId = Number(item.id);
    highlightSegment(item.id);
    const row = $(`#semanticTable .semantic-row.data[data-segment-id="${Number(item.id)}"]`);
    const english = row?.children[2]?.textContent?.trim();
    if (english) elements.nowPlaying.textContent = english;
    elements.ttsAudio.src = `${item.url}?t=${Date.now()}`;
    elements.audioBadge.textContent = "正在播放";
    elements.playButton.textContent = "Ⅱ";
    renderQueue(runtime.currentSnapshot || { rows: [], played_segment_ids: [] });
    try {
      await elements.ttsAudio.play();
    } catch (error) {
      runtime.audioQueue.unshift(item);
      runtime.playingSegmentId = null;
      elements.audioBadge.textContent = "点击播放";
      elements.playButton.textContent = "▶";
      showToast("浏览器已拦截自动播放，点击播放按钮即可继续。", 3500);
    }
  }

  function playNextAudio() {
    if (runtime.playingSegmentId || !runtime.audioQueue.length) return;
    const item = runtime.audioQueue.shift();
    playAudioItem(item);
  }

  elements.ttsAudio.addEventListener("timeupdate", () => {
    const duration = elements.ttsAudio.duration || 0;
    const current = elements.ttsAudio.currentTime || 0;
    elements.trackFill.style.width = duration ? `${Math.min(100, current / duration * 100)}%` : "0%";
    elements.duration.textContent = `${formatDuration(current)} / ${formatDuration(duration)}`;
  });

  elements.ttsAudio.addEventListener("ended", () => {
    runtime.playingSegmentId = null;
    elements.playButton.textContent = "▶";
    elements.audioBadge.textContent = runtime.audioQueue.length ? "继续播放" : "等待播放";
    renderQueue(runtime.currentSnapshot || { rows: [], played_segment_ids: [] });
    if (window.V4Settings?.get().autoplay ?? true) playNextAudio();
  });

  window.addEventListener("v4settingschange", (event) => {
    setPhase(runtime.phase);
    renderRows(runtime.currentSnapshot?.rows || []);
    renderQueue(runtime.currentSnapshot || { rows: [], played_segment_ids: [] });
    if (event.detail?.autoplay && runtime.audioQueue.length && elements.ttsAudio.paused && !runtime.playingSegmentId) {
      playNextAudio();
    }
  });

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return "00:00";
    const total = Math.max(0, Math.round(seconds));
    return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  }

  async function warmup() {
    setPhase("loading");
    try {
      const result = await api("/api/models/warmup", { method: "POST" });
      runtime.modelReady = true;
      setPhase("ready", `模型就绪 · 启动耗时 ${Number(result.elapsed || 0).toFixed(1)} 秒`);
      showToast("本地模型已全部就绪");
      await refreshSystem();
    } catch (error) {
      setPhase("error", error.message);
      showToast(error.message, 5000);
    }
  }

  async function createSession() {
    const snapshot = await api("/api/sessions", { method: "POST" });
    applySnapshot(snapshot);
  }

  function downsample(input, inputRate, outputRate = 16000) {
    if (inputRate === outputRate) return new Float32Array(input);
    const ratio = inputRate / outputRate;
    const length = Math.max(1, Math.round(input.length / ratio));
    const output = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      const start = Math.floor(index * ratio);
      const end = Math.min(input.length, Math.floor((index + 1) * ratio));
      let sum = 0;
      for (let source = start; source < end; source += 1) sum += input[source];
      output[index] = sum / Math.max(1, end - start);
    }
    return output;
  }

  function appendSamples(samples) {
    runtime.pendingSamples.push(samples);
    runtime.pendingSampleCount += samples.length;
    const blockSize = 6400;
    while (runtime.pendingSampleCount >= blockSize) sendPendingSamples(blockSize);
  }

  function takeSamples(sampleCount) {
    const output = new Float32Array(sampleCount);
    let offset = 0;
    while (offset < sampleCount && runtime.pendingSamples.length) {
      const first = runtime.pendingSamples[0];
      const count = Math.min(first.length, sampleCount - offset);
      output.set(first.subarray(0, count), offset);
      offset += count;
      if (count === first.length) runtime.pendingSamples.shift();
      else runtime.pendingSamples[0] = first.subarray(count);
    }
    runtime.pendingSampleCount -= offset;
    return output.subarray(0, offset);
  }

  function floatToInt16(samples) {
    const output = new Int16Array(samples.length);
    for (let index = 0; index < samples.length; index += 1) {
      const value = Math.max(-1, Math.min(1, samples[index]));
      output[index] = value < 0 ? value * 32768 : value * 32767;
    }
    return output;
  }

  function sendPendingSamples(sampleCount) {
    const samples = takeSamples(Math.min(sampleCount, runtime.pendingSampleCount));
    if (!samples.length || !runtime.sessionId) return;
    const pcm = floatToInt16(samples);
    runtime.sendChain = runtime.sendChain.then(async () => {
      const snapshot = await api(`/api/sessions/${runtime.sessionId}/chunks`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: pcm.buffer,
      });
      applySnapshot(snapshot);
    }).catch((error) => {
      showToast(`音频处理失败：${error.message}`, 4500);
    });
  }

  async function startMicrophone() {
    await createSession();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    runtime.stream = stream;
    runtime.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    await runtime.audioContext.resume();
    runtime.mediaSource = runtime.audioContext.createMediaStreamSource(stream);
    runtime.processor = runtime.audioContext.createScriptProcessor(4096, 1, 1);
    runtime.silentGain = runtime.audioContext.createGain();
    runtime.visualAnalyser = runtime.audioContext.createAnalyser();
    runtime.silentGain.gain.value = 0;
    runtime.processor.onaudioprocess = (event) => {
      if (runtime.phase !== "live") return;
      appendSamples(downsample(event.inputBuffer.getChannelData(0), runtime.audioContext.sampleRate));
    };
    runtime.mediaSource.connect(runtime.processor);
    runtime.mediaSource.connect(runtime.visualAnalyser);
    runtime.processor.connect(runtime.silentGain);
    runtime.visualAnalyser.connect(runtime.silentGain);
    runtime.silentGain.connect(runtime.audioContext.destination);
    startWaveform(runtime.visualAnalyser);
    setPhase("live", "麦克风正在持续输入");
    elements.inputDetail.textContent = "每 0.4 秒推送一个音频块";
    showToast("麦克风同传已开始");
  }

  async function startFile() {
    if (!runtime.selectedFile) {
      elements.wavFileInput.click();
      showToast("请先选择一个 WAV 演讲音频");
      return;
    }
    const form = new FormData();
    form.append("file", runtime.selectedFile, runtime.selectedFile.name);
    const snapshot = await api("/api/file-sessions", { method: "POST", body: form });
    applySnapshot(snapshot);
    if (runtime.sourceObjectUrl) URL.revokeObjectURL(runtime.sourceObjectUrl);
    runtime.sourceObjectUrl = URL.createObjectURL(runtime.selectedFile);
    elements.sourceAudio.src = runtime.sourceObjectUrl;
    await prepareFileWaveform();
    await elements.sourceAudio.play().catch(() => showToast("点击页面后源音频即可同步播放。", 3200));
    setPhase("live", "WAV 正在按时间轴流式推送");
    startPolling();
    showToast("文件流式同传已开始");
  }

  async function startInterpretation() {
    runtime.mode = selectedMode();
    clearLiveOutput();
    runtime.pendingSamples = [];
    runtime.pendingSampleCount = 0;
    runtime.audioQueue = [];
    runtime.queuedSegmentIds.clear();
    try {
      if (runtime.mode === "file") await startFile();
      else await startMicrophone();
    } catch (error) {
      await releaseMicrophone();
      setPhase("ready", `启动失败：${error.message}`);
      showToast(error.message, 5000);
    }
  }

  async function releaseMicrophone() {
    resetWaveform();
    if (runtime.processor) runtime.processor.disconnect();
    if (runtime.mediaSource) runtime.mediaSource.disconnect();
    if (runtime.visualAnalyser) runtime.visualAnalyser.disconnect();
    if (runtime.silentGain) runtime.silentGain.disconnect();
    if (runtime.stream) runtime.stream.getTracks().forEach((track) => track.stop());
    if (runtime.audioContext && runtime.audioContext.state !== "closed") await runtime.audioContext.close();
    runtime.processor = runtime.mediaSource = runtime.silentGain = runtime.stream = runtime.audioContext = null;
  }

  function stopPolling() {
    window.clearInterval(runtime.pollTimer);
    runtime.pollTimer = null;
  }

  function startPolling() {
    stopPolling();
    runtime.pollTimer = window.setInterval(async () => {
      if (!runtime.sessionId) return;
      try {
        const snapshot = await api(`/api/sessions/${runtime.sessionId}`);
        applySnapshot(snapshot);
        if (snapshot.file_status === "DONE" || snapshot.file_status === "ERROR") {
          stopPolling();
          elements.sourceAudio.pause();
          resetWaveform();
          setPhase("ready", snapshot.file_status === "DONE" ? "文件同传已完成" : "文件处理已停止");
          await refreshResults();
        }
      } catch (error) {
        stopPolling();
        showToast(error.message, 4200);
      }
    }, 500);
  }

  async function stopInterpretation() {
    setPhase("stopping");
    try {
      if (runtime.mode === "file") {
        elements.sourceAudio.pause();
        resetWaveform();
        await api(`/api/sessions/${runtime.sessionId}/stop`, { method: "POST" });
        stopPolling();
      } else {
        await releaseMicrophone();
        if (runtime.pendingSampleCount) sendPendingSamples(runtime.pendingSampleCount);
        await runtime.sendChain;
        let snapshot = await api(`/api/sessions/${runtime.sessionId}/finalize`, { method: "POST" });
        applySnapshot(snapshot);
        for (let attempt = 0; attempt < 120 && snapshot.finalization_status === "RUNNING"; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 350));
          snapshot = await api(`/api/sessions/${runtime.sessionId}`);
          applySnapshot(snapshot);
        }
      }
      setPhase("ready", "本次同传已结束 · 模型保持就绪");
      elements.inputDetail.textContent = "可再次开始";
      showToast("同传已停止，结果已保留");
      await refreshResults();
    } catch (error) {
      setPhase("ready", `结束处理异常：${error.message}`);
      showToast(error.message, 5000);
    }
  }

  async function ignitionAction(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
    if (runtime.phase === "loading" || runtime.phase === "stopping") return;
    if (!runtime.modelReady) return warmup();
    if (runtime.phase === "live") return stopInterpretation();
    return startInterpretation();
  }

  elements.ignition.addEventListener("click", ignitionAction, true);

  elements.selectWavButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    elements.wavFileInput.click();
  }, true);

  elements.wavFileInput.addEventListener("change", () => {
    const file = elements.wavFileInput.files?.[0];
    if (!file) return;
    runtime.selectedFile = file;
    elements.selectWavButton.textContent = file.name;
    elements.inputDetail.textContent = `已选择 ${file.name}`;
    elements.uploadPanel.querySelector("span").textContent = "将与源音频同步，按真实时间轴持续推送。";
    showToast(`已选择 ${file.name}`);
  });

  document.querySelectorAll(".segmented button").forEach((button) => {
    button.addEventListener("click", () => {
      runtime.mode = button.dataset.mode === "file" ? "file" : "mic";
    }, true);
  });

  elements.semanticTable.addEventListener("click", (event) => {
    const row = event.target.closest(".semantic-row.data[data-segment-id]");
    if (!row || !runtime.sessionId) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const segmentId = Number(row.dataset.segmentId);
    runtime.audioQueue = runtime.audioQueue.filter((item) => item.id !== segmentId);
    runtime.playingSegmentId = null;
    elements.ttsAudio.pause();
    playAudioItem({ id: segmentId, url: `/api/sessions/${runtime.sessionId}/audio/${segmentId}` });
  }, true);

  elements.playButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    if (!elements.ttsAudio.paused) {
      elements.ttsAudio.pause();
      elements.playButton.textContent = "▶";
      elements.audioBadge.textContent = "已暂停";
      return;
    }
    if (elements.ttsAudio.src) {
      elements.ttsAudio.play().catch(() => showToast("暂时没有可播放的英文语音。"));
      elements.playButton.textContent = "Ⅱ";
      elements.audioBadge.textContent = "正在播放";
    } else {
      playNextAudio();
      if (!runtime.audioQueue.length) showToast("英文语音生成后会自动进入播放队列。", 2800);
    }
  }, true);

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  async function refreshResults() {
    try {
      const result = await api("/api/results");
      if (!result.files?.length) return;
      elements.backendFileList.innerHTML = result.files.map((file) => `<div class="file-row"><span class="file-name">${escapeHtml(file.name)}</span><span class="file-meta">本地输出</span><span class="file-meta">${formatBytes(file.size)}</span><button class="small-button" type="button" disabled>已生成</button></div>`).join("");
    } catch (error) {
      showToast(`结果读取失败：${error.message}`);
    }
  }

  async function refreshSystem() {
    try {
      const result = await api("/api/system");
      const states = elements.systemModuleGrid.querySelectorAll(".module-state");
      const ready = result.models?.status === "ready";
      states.forEach((state) => state.lastChild.textContent = ready ? "已就绪" : "待启动");
    } catch (error) {
      showToast(`系统状态读取失败：${error.message}`);
    }
  }

  document.querySelector('[data-page="results"]')?.addEventListener("click", refreshResults);
  document.querySelector('[data-page="system"]')?.addEventListener("click", refreshSystem);

  window.addEventListener("beforeunload", () => {
    stopPolling();
    resetWaveform();
    if (runtime.stream) runtime.stream.getTracks().forEach((track) => track.stop());
    if (runtime.fileAudioContext && runtime.fileAudioContext.state !== "closed") runtime.fileAudioContext.close();
  });

  resetWaveform();
  clearLiveOutput();

  api("/api/health").then((health) => {
    runtime.modelReady = health.models?.status === "ready";
    if (runtime.modelReady) setPhase("ready", "模型已在本地服务中就绪");
    else setPhase("idle");
  }).catch(() => {
    setPhase("error", "无法连接本地同传服务");
    showToast("无法连接本地同传服务，请先运行启动脚本。", 5000);
  });
})();
