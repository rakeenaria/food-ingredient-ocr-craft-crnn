// Label capture: camera / upload / sample -> crop & rotate -> submit.
// Everything happens in the browser; Python receives one final image.
//
// Data from Python:
//   samples: [{ id, label, thumb }]           sample thumbnails
//   image:   { token, src, kind } | null      an image to open in the editor
//   notice:  { token, text } | null           a message to show (re-enables submit)
// Triggers to Python:
//   sample: <sample id>
//   submit: { image, width, height, source, edited }

const instances = new WeakMap();

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const PASSTHROUGH_MAX_BYTES = 8 * 1024 * 1024;
const MAX_OUTPUT_SIDE = 2560;
const PREVIEW_MAX_SIDE = 1600;
const JPEG_QUALITY = 0.92;
const MIN_CROP_CSS = 24;
const HANDLE_HIT_CSS = 30;
const HANDLE_DRAW_CSS = 11;

const IDLE_TEXT = "Point your camera at the ingredient list.<br />Good light, no glare, text in focus.";

const CURSORS = {
  n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
  ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize",
  move: "move", new: "crosshair",
};

export default function (component) {
  const { parentElement, data, setTriggerValue } = component;
  let inst = instances.get(parentElement);
  if (!inst) {
    inst = createInstance(parentElement);
    instances.set(parentElement, inst);
  }
  inst.setTriggerValue = setTriggerValue;
  inst.update(data || {});
  // Release the camera when Streamlit removes the component.
  return () => inst.stopCamera();
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function createInstance(root) {
  const q = (selector) => root.querySelector(selector);
  const qa = (selector) => Array.from(root.querySelectorAll(selector));

  const el = {
    steps: qa("[data-step]"),
    source: q('[data-stage="source"]'),
    editor: q('[data-stage="editor"]'),
    tabs: qa("[data-tab]"),
    panels: qa("[data-panel]"),
    viewfinder: q(".viewfinder"),
    video: q("video"),
    startButton: q('[data-action="start-camera"]'),
    idleText: q(".camera-idle p"),
    cameraControls: q(".camera-controls"),
    switchButton: q('[data-action="switch-camera"]'),
    fileInput: q('input[type="file"]'),
    drop: q(".drop"),
    samples: q(".samples"),
    sampleList: q(".sample-list"),
    canvasWrap: q(".canvas-wrap"),
    canvas: q(".crop-canvas"),
    angle: q(".straighten input"),
    angleOut: q(".straighten output"),
    submitButton: q('[data-action="submit"]'),
    message: q(".message"),
  };

  const s = {
    stream: null,
    cameraRequest: 0,
    facing: "environment",
    image: null, // full-resolution source (HTMLImageElement or canvas)
    preview: null, // downscaled copy used for on-screen drawing
    imageW: 0,
    imageH: 0,
    kind: null,
    original: null, // original data URL, sent unchanged when the user edits nothing
    quarter: 0,
    fine: 0,
    crop: null, // {x, y, w, h} in rotated-image pixels
    drag: null,
    scale: 1,
    dpr: 1,
    frame: 0,
    wrapWidth: 0,
    imageToken: null,
    noticeToken: null,
    samplesSignature: null,
  };

  const inst = { setTriggerValue: () => {} };

  // ---------------------------------------------------------------- stage
  function setStage(stage) {
    el.source.hidden = stage !== "source";
    el.editor.hidden = stage !== "editor";
    const active = stage === "source" ? 1 : 2;
    for (const item of el.steps) {
      const step = Number(item.dataset.step);
      item.classList.toggle("is-active", step === active);
      item.classList.toggle("is-done", step < active);
    }
  }

  function showMessage(text) {
    el.message.textContent = text || "";
    el.message.hidden = !text;
    if (text) el.message.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // ----------------------------------------------------------------- tabs
  function selectTab(name) {
    for (const tab of el.tabs) tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
    for (const panel of el.panels) panel.hidden = panel.dataset.panel !== name;
    if (name !== "camera") stopCamera();
    showMessage("");
  }
  for (const tab of el.tabs) tab.addEventListener("click", () => selectTab(tab.dataset.tab));

  // --------------------------------------------------------------- camera
  function setStarting(starting) {
    el.startButton.disabled = starting;
    el.startButton.textContent = starting ? "Starting camera…" : "Start camera";
  }

  async function startCamera() {
    showMessage("");
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showMessage("The camera needs a secure (https) connection and a supported browser. Use Upload image instead.");
      return;
    }
    stopCamera();
    const request = s.cameraRequest;
    setStarting(true);
    el.idleText.textContent = "Allow camera access when your browser asks.";
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: s.facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
    } catch (error) {
      if (request !== s.cameraRequest) return; // the visitor moved on meanwhile
      setStarting(false);
      el.idleText.innerHTML = IDLE_TEXT;
      const name = error && error.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        showMessage("Camera access was blocked. Allow the camera for this site in your browser settings, or use Upload image.");
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        showMessage("No camera was found on this device. Use Upload image instead.");
      } else if (name === "NotReadableError") {
        showMessage("The camera is in use by another app. Close it and try again, or use Upload image.");
      } else {
        showMessage("The camera could not be started. Use Upload image instead.");
      }
      return;
    }
    if (request !== s.cameraRequest) {
      // Permission arrived after the visitor switched tabs or moved on: release it.
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    setStarting(false);
    el.idleText.innerHTML = IDLE_TEXT;
    s.stream = stream;
    el.video.srcObject = s.stream;
    try {
      await el.video.play();
    } catch (_) {
      // Autoplay is allowed for muted inline video; ignore rare play() rejections.
    }
    el.viewfinder.classList.add("is-live");
    el.cameraControls.hidden = false;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      el.switchButton.hidden = devices.filter((d) => d.kind === "videoinput").length < 2;
    } catch (_) {
      el.switchButton.hidden = true;
    }
  }

  function stopCamera() {
    s.cameraRequest += 1; // invalidates any permission request still pending
    setStarting(false);
    el.idleText.innerHTML = IDLE_TEXT;
    if (s.stream) {
      for (const track of s.stream.getTracks()) track.stop();
      s.stream = null;
    }
    el.video.srcObject = null;
    el.viewfinder.classList.remove("is-live");
    el.viewfinder.style.aspectRatio = "";
    el.cameraControls.hidden = true;
  }

  function capture() {
    const width = el.video.videoWidth;
    const height = el.video.videoHeight;
    if (!width || !height) {
      showMessage("The camera is not ready yet. Try again in a moment.");
      return;
    }
    const frame = document.createElement("canvas");
    frame.width = width;
    frame.height = height;
    frame.getContext("2d").drawImage(el.video, 0, 0, width, height);
    stopCamera();
    openEditor(frame, width, height, "camera", null);
  }

  el.video.addEventListener("loadedmetadata", () => {
    if (el.video.videoWidth && el.video.videoHeight) {
      el.viewfinder.style.aspectRatio = `${el.video.videoWidth} / ${el.video.videoHeight}`;
    }
  });
  el.startButton.addEventListener("click", startCamera);
  q('[data-action="capture"]').addEventListener("click", capture);
  el.switchButton.addEventListener("click", () => {
    s.facing = s.facing === "environment" ? "user" : "environment";
    startCamera();
  });

  // -------------------------------------------------------------- uploads
  function readFile(file) {
    showMessage("");
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      showMessage(`This image is ${(file.size / 1048576).toFixed(1)} MB. The limit is 20 MB.`);
      return;
    }
    if (file.type && !/^image\/(jpeg|png|webp)$/.test(file.type)) {
      showMessage("Please choose a JPG, PNG or WebP image.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => loadDataUrl(String(reader.result), "upload", file.size);
    reader.onerror = () => showMessage("This file could not be read. Please try another image.");
    reader.readAsDataURL(file);
  }

  function loadDataUrl(dataUrl, kind, byteSize) {
    const image = new Image();
    image.onload = () => {
      const width = image.naturalWidth;
      const height = image.naturalHeight;
      if (width < 32 || height < 32) {
        showMessage("This image is too small to read. Use a photo at least 32 pixels on each side.");
        return;
      }
      const passThrough =
        byteSize <= PASSTHROUGH_MAX_BYTES &&
        Math.max(width, height) <= MAX_OUTPUT_SIDE &&
        /^data:image\/(jpeg|png|webp)/.test(dataUrl);
      openEditor(image, width, height, kind, passThrough ? dataUrl : null);
    };
    image.onerror = () => showMessage("This image format is not supported. Please use JPG, PNG or WebP.");
    image.src = dataUrl;
  }

  el.fileInput.addEventListener("change", () => {
    const file = el.fileInput.files && el.fileInput.files[0];
    el.fileInput.value = "";
    readFile(file);
  });
  for (const type of ["dragenter", "dragover"]) {
    el.drop.addEventListener(type, (event) => {
      event.preventDefault();
      el.drop.classList.add("is-over");
    });
  }
  el.drop.addEventListener("dragleave", () => el.drop.classList.remove("is-over"));
  el.drop.addEventListener("drop", (event) => {
    event.preventDefault();
    el.drop.classList.remove("is-over");
    readFile(event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
  });

  // -------------------------------------------------------------- samples
  function renderSamples(samples) {
    const signature = JSON.stringify(samples.map((sample) => sample.id));
    if (signature !== s.samplesSignature) {
      s.samplesSignature = signature;
      el.sampleList.replaceChildren(
        ...samples.map((sample) => {
          const button = document.createElement("button");
          button.type = "button";
          button.title = sample.label;
          button.setAttribute("aria-label", `Use ${sample.label}`);
          const thumb = document.createElement("img");
          thumb.src = sample.thumb;
          thumb.alt = "";
          button.appendChild(thumb);
          button.addEventListener("click", () => {
            showMessage("");
            for (const other of el.sampleList.querySelectorAll("button")) other.disabled = true;
            inst.setTriggerValue("sample", sample.id);
          });
          return button;
        }),
      );
    }
    for (const button of el.sampleList.querySelectorAll("button")) button.disabled = false;
    el.samples.hidden = samples.length === 0;
  }

  // --------------------------------------------------------------- editor
  function makePreview(image, width, height) {
    const k = Math.min(1, PREVIEW_MAX_SIDE / Math.max(width, height));
    if (k === 1) return image;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * k);
    canvas.height = Math.round(height * k);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  function openEditor(image, width, height, kind, original) {
    stopCamera();
    s.image = image;
    s.preview = makePreview(image, width, height);
    s.imageW = width;
    s.imageH = height;
    s.kind = kind;
    s.original = original;
    s.quarter = 0;
    s.fine = 0;
    s.crop = { x: 0, y: 0, w: width, h: height };
    el.angle.value = "0";
    el.angleOut.textContent = "0°";
    el.submitButton.disabled = false;
    el.submitButton.textContent = "Read ingredients";
    showMessage("");
    setStage("editor");
    requestAnimationFrame(() => {
      layoutCanvas();
      render();
    });
  }

  function closeEditor() {
    s.image = null;
    s.preview = null;
    s.original = null;
    s.crop = null;
    s.drag = null;
    showMessage("");
    setStage("source");
  }

  function angleRadians() {
    return ((s.quarter * 90 + s.fine) * Math.PI) / 180;
  }

  function rotatedSize() {
    const t = angleRadians();
    const cos = Math.abs(Math.cos(t));
    const sin = Math.abs(Math.sin(t));
    return { w: s.imageW * cos + s.imageH * sin, h: s.imageW * sin + s.imageH * cos };
  }

  // Draw the rotated image so rotated-space point (offsetX, offsetY) lands at the canvas origin.
  function drawRotated(ctx, source, scale, offsetX, offsetY) {
    const { w, h } = rotatedSize();
    ctx.save();
    ctx.scale(scale, scale);
    ctx.translate(w / 2 - offsetX, h / 2 - offsetY);
    ctx.rotate(angleRadians());
    ctx.drawImage(source, -s.imageW / 2, -s.imageH / 2, s.imageW, s.imageH);
    ctx.restore();
  }

  function layoutCanvas() {
    if (!s.image) return;
    const { w, h } = rotatedSize();
    s.wrapWidth = el.canvasWrap.clientWidth;
    const maxWidth = Math.max(160, s.wrapWidth - 2);
    const maxHeight = Math.max(240, Math.min(window.innerHeight * 0.62, 760));
    s.scale = Math.min(maxWidth / w, maxHeight / h);
    const cssWidth = Math.max(1, Math.floor(w * s.scale));
    const cssHeight = Math.max(1, Math.floor(h * s.scale));
    s.dpr = Math.min(window.devicePixelRatio || 1, 3);
    el.canvas.style.width = `${cssWidth}px`;
    el.canvas.style.height = `${cssHeight}px`;
    el.canvas.width = Math.round(cssWidth * s.dpr);
    el.canvas.height = Math.round(cssHeight * s.dpr);
  }

  function handlePoints(rect) {
    const { x, y, w, h } = rect;
    return [
      ["nw", x, y], ["n", x + w / 2, y], ["ne", x + w, y], ["e", x + w, y + h / 2],
      ["se", x + w, y + h], ["s", x + w / 2, y + h], ["sw", x, y + h], ["w", x, y + h / 2],
    ];
  }

  function render() {
    s.frame = 0;
    if (!s.image || !s.crop) return;
    const ctx = el.canvas.getContext("2d");
    const k = s.scale * s.dpr;
    const { w, h } = rotatedSize();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, el.canvas.width, el.canvas.height);
    drawRotated(ctx, s.preview, k, 0, 0);

    const c = s.crop;
    ctx.save();
    ctx.scale(k, k);
    ctx.fillStyle = "rgba(255, 255, 255, 0.74)";
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.rect(c.x, c.y, c.w, c.h);
    ctx.fill("evenodd");
    ctx.restore();

    const frame = { x: c.x * k, y: c.y * k, w: c.w * k, h: c.h * k };
    ctx.lineWidth = 2 * s.dpr;
    ctx.strokeStyle = "#000000";
    ctx.strokeRect(frame.x, frame.y, frame.w, frame.h);
    const size = HANDLE_DRAW_CSS * s.dpr;
    for (const [, hx, hy] of handlePoints(frame)) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(hx - size / 2 - s.dpr, hy - size / 2 - s.dpr, size + 2 * s.dpr, size + 2 * s.dpr);
      ctx.fillStyle = "#000000";
      ctx.fillRect(hx - size / 2, hy - size / 2, size, size);
    }
  }

  function scheduleRender() {
    if (!s.frame) s.frame = requestAnimationFrame(render);
  }

  function setRotation(quarter, fine, keepCrop) {
    const before = rotatedSize();
    const relative = s.crop && keepCrop
      ? { x: s.crop.x / before.w, y: s.crop.y / before.h, w: s.crop.w / before.w, h: s.crop.h / before.h }
      : null;
    s.quarter = ((quarter % 4) + 4) % 4;
    s.fine = fine;
    const after = rotatedSize();
    s.crop = relative
      ? { x: relative.x * after.w, y: relative.y * after.h, w: relative.w * after.w, h: relative.h * after.h }
      : { x: 0, y: 0, w: after.w, h: after.h };
    layoutCanvas();
    render();
  }

  function canvasPoint(event) {
    const box = el.canvas.getBoundingClientRect();
    return { cx: event.clientX - box.left, cy: event.clientY - box.top };
  }

  function hitTest(cx, cy) {
    const c = s.crop;
    const rect = { x: c.x * s.scale, y: c.y * s.scale, w: c.w * s.scale, h: c.h * s.scale };
    const half = HANDLE_HIT_CSS / 2;
    for (const [name, hx, hy] of handlePoints(rect)) {
      if (Math.abs(cx - hx) <= half && Math.abs(cy - hy) <= half) return name;
    }
    if (cx >= rect.x && cx <= rect.x + rect.w && cy >= rect.y && cy <= rect.y + rect.h) return "move";
    return "new";
  }

  el.canvas.addEventListener("pointerdown", (event) => {
    if (!s.image || !s.crop) return;
    event.preventDefault();
    const { cx, cy } = canvasPoint(event);
    const { w, h } = rotatedSize();
    s.drag = {
      mode: hitTest(cx, cy),
      startX: clamp(cx / s.scale, 0, w),
      startY: clamp(cy / s.scale, 0, h),
      crop: { ...s.crop },
      pointerId: event.pointerId,
    };
    el.canvas.setPointerCapture(event.pointerId);
  });

  el.canvas.addEventListener("pointermove", (event) => {
    if (!s.image || !s.crop) return;
    const { cx, cy } = canvasPoint(event);
    if (!s.drag) {
      el.canvas.style.cursor = CURSORS[hitTest(cx, cy)];
      return;
    }
    const { w: W, h: H } = rotatedSize();
    const px = clamp(cx / s.scale, 0, W);
    const py = clamp(cy / s.scale, 0, H);
    const d = s.drag;
    const min = MIN_CROP_CSS / s.scale;
    let { x, y, w, h } = d.crop;
    if (d.mode === "move") {
      x = clamp(x + px - d.startX, 0, W - w);
      y = clamp(y + py - d.startY, 0, H - h);
    } else if (d.mode === "new") {
      w = Math.abs(px - d.startX);
      h = Math.abs(py - d.startY);
      if (w < min || h < min) return; // ignore taps and tiny drags
      x = Math.min(d.startX, px);
      y = Math.min(d.startY, py);
    } else {
      const dx = px - d.startX;
      const dy = py - d.startY;
      let x1 = x;
      let y1 = y;
      let x2 = x + w;
      let y2 = y + h;
      if (d.mode.includes("w")) x1 = clamp(x1 + dx, 0, x2 - min);
      if (d.mode.includes("e")) x2 = clamp(x2 + dx, x1 + min, W);
      if (d.mode.includes("n")) y1 = clamp(y1 + dy, 0, y2 - min);
      if (d.mode.includes("s")) y2 = clamp(y2 + dy, y1 + min, H);
      x = x1;
      y = y1;
      w = x2 - x1;
      h = y2 - y1;
    }
    s.crop = { x, y, w, h };
    scheduleRender();
  });

  function endDrag() {
    if (!s.drag) return;
    try {
      el.canvas.releasePointerCapture(s.drag.pointerId);
    } catch (_) {
      // Capture may already be released.
    }
    s.drag = null;
  }
  el.canvas.addEventListener("pointerup", endDrag);
  el.canvas.addEventListener("pointercancel", endDrag);

  q('[data-action="rotate-left"]').addEventListener("click", () => setRotation(s.quarter - 1, s.fine, false));
  q('[data-action="rotate-right"]').addEventListener("click", () => setRotation(s.quarter + 1, s.fine, false));
  el.angle.addEventListener("input", () => {
    const value = Number(el.angle.value);
    el.angleOut.textContent = `${value > 0 ? "+" : ""}${value}°`;
    setRotation(s.quarter, value, true);
  });
  q('[data-action="reset"]').addEventListener("click", () => {
    el.angle.value = "0";
    el.angleOut.textContent = "0°";
    setRotation(0, 0, false);
  });
  q('[data-action="back"]').addEventListener("click", () => {
    const retake = s.kind === "camera";
    closeEditor();
    // Back from a camera photo means "retake": permission is already granted.
    if (retake) startCamera();
  });

  new ResizeObserver(() => {
    if (el.editor.hidden || !s.image) return;
    if (el.canvasWrap.clientWidth === s.wrapWidth) return;
    layoutCanvas();
    render();
  }).observe(el.canvasWrap);

  // --------------------------------------------------------------- submit
  function exportImage() {
    const { w, h } = rotatedSize();
    const c = s.crop;
    const untouched =
      s.quarter === 0 && s.fine === 0 &&
      c.x <= 0.5 && c.y <= 0.5 && Math.abs(c.w - w) <= 1 && Math.abs(c.h - h) <= 1;
    if (untouched && s.original) {
      return { image: s.original, width: s.imageW, height: s.imageH, edited: false };
    }
    const k = Math.min(1, MAX_OUTPUT_SIDE / Math.max(c.w, c.h));
    const width = Math.max(1, Math.round(c.w * k));
    const height = Math.max(1, Math.round(c.h * k));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.imageSmoothingQuality = "high";
    drawRotated(ctx, s.image, k, c.x, c.y);
    return { image: canvas.toDataURL("image/jpeg", JPEG_QUALITY), width, height, edited: true };
  }

  el.submitButton.addEventListener("click", () => {
    if (!s.image || el.submitButton.disabled) return;
    let output;
    try {
      output = exportImage();
    } catch (_) {
      showMessage("The photo could not be prepared. Try a smaller crop or another image.");
      return;
    }
    el.submitButton.disabled = true;
    el.submitButton.textContent = "Sending…";
    inst.setTriggerValue("submit", { ...output, source: s.kind });
  });

  // ------------------------------------------------------------- initial
  setStage("source");
  // Default to Upload on devices without a camera (most desktops).
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    navigator.mediaDevices
      .enumerateDevices()
      .then((devices) => {
        if (!devices.some((d) => d.kind === "videoinput")) selectTab("upload");
      })
      .catch(() => {});
  } else {
    selectTab("upload");
  }

  inst.stopCamera = stopCamera;
  inst.update = (data) => {
    renderSamples(Array.isArray(data.samples) ? data.samples : []);
    const image = data.image;
    if (image && image.src && image.token !== s.imageToken) {
      s.imageToken = image.token;
      const base64Length = image.src.length - image.src.indexOf(",") - 1;
      loadDataUrl(image.src, image.kind || "sample", Math.floor(base64Length * 0.75));
    }
    const notice = data.notice;
    if (notice && notice.token !== s.noticeToken) {
      s.noticeToken = notice.token;
      showMessage(notice.text || "");
      el.submitButton.disabled = false;
      el.submitButton.textContent = "Read ingredients";
    }
  };
  return inst;
}
