// The label text box: the text as read, with words to check (dotted underline)
// and words corrected automatically (yellow). Hovering or tapping a corrected
// word shows what was read, with Undo. Copy, download and editing all happen in
// the browser.
//
// Data from Python:
//   text       the text to show
//   marks      [{ start, end, kind: "low" | "fixed" | "undone", id, original, replacement, tip }]
//   rejected   ids of the words whose correction was undone
//   editing    show the editor instead of the text
//   editText   the text to start editing with; editToken changes for each new edit
//   edited     the text is the user's own edit (offers "Undo my edits")
//   fileName   name of the downloaded file
// Sent back to Python:
//   state   rejected: [id, ...]            corrections the user undid
//   state   draft: { token, text }         the editor's text when it loses focus
//   trigger revert                         "Undo my edits"
// The draft is state, not a trigger: Streamlit keeps the latest state even when
// another widget (the Edit toggle) asks for a rerun at the same moment.

const instances = new WeakMap();

export default function (component) {
  const { parentElement, data, setStateValue, setTriggerValue } = component;
  let inst = instances.get(parentElement);
  if (!inst) {
    inst = createInstance(parentElement);
    instances.set(parentElement, inst);
  }
  inst.setStateValue = setStateValue;
  inst.setTriggerValue = setTriggerValue;
  inst.update(data || {});
  return () => inst.closePop();
}

function createInstance(root) {
  const q = (selector) => root.querySelector(selector);
  const box = q(".box");
  const textEl = q(".text");
  const editor = q(".editor");
  const hint = q(".edit-hint");
  const foot = q(".foot");
  const pop = q(".pop");
  const popText = q(".pop-text");
  const popAction = q(".pop-action");
  const copyButton = q('[data-action="copy"]');
  const downloadButton = q('[data-action="download"]');
  const revertButton = q('[data-action="revert"]');
  const live = q(".sr");

  const inst = { setStateValue: () => {}, setTriggerValue: () => {} };
  let data = {};
  let signature = null;
  let editToken = null;
  let rejected = new Set();
  let open = null; // { el, mark, pinned }
  let hoverTimer = 0;
  let closeTimer = 0;
  let copyTimer = 0;

  // ------------------------------------------------------------------ text
  function renderText() {
    closePop();
    const text = String(data.text || "");
    const marks = (Array.isArray(data.marks) ? data.marks : [])
      .filter((mark) => mark && Number(mark.end) > Number(mark.start))
      .sort((a, b) => a.start - b.start);
    const nodes = [];
    let cursor = 0;
    for (const mark of marks) {
      const start = Number(mark.start);
      const end = Number(mark.end);
      if (start < cursor || end > text.length) continue;
      if (start > cursor) nodes.push(document.createTextNode(text.slice(cursor, start)));
      const piece = text.slice(start, end);
      const span = document.createElement("span");
      span.textContent = piece;
      if (mark.kind === "fixed" || mark.kind === "undone") {
        span.className = mark.kind === "fixed" ? "fix" : "fix is-undone";
        span.tabIndex = 0;
        span.setAttribute("role", "button");
        span.setAttribute("aria-haspopup", "dialog");
        span.setAttribute("aria-expanded", "false");
        span.setAttribute(
          "aria-label",
          mark.kind === "fixed" ? `${piece}, corrected from ${mark.original}` : `${piece}, could be ${mark.replacement}`,
        );
        span._mark = mark;
      } else {
        span.className = "low";
        if (mark.tip) span.title = mark.tip;
      }
      nodes.push(span);
      cursor = end;
    }
    if (cursor < text.length) nodes.push(document.createTextNode(text.slice(cursor)));
    textEl.replaceChildren(...nodes);
  }

  // --------------------------------------------------------------- popover
  function isFixed(el) {
    return !el.classList.contains("is-undone");
  }

  function showPop(el, pinned) {
    clearTimeout(hoverTimer);
    clearTimeout(closeTimer);
    if (open && open.el !== el) open.el.setAttribute("aria-expanded", "false");
    const mark = el._mark;
    const fixed = isFixed(el);
    const word = document.createElement(fixed ? "code" : "b");
    word.textContent = fixed ? mark.original : `“${mark.replacement}”`;
    popText.replaceChildren(document.createTextNode(fixed ? "Read as " : "Could be "), word);
    popAction.textContent = fixed ? "Undo" : "Use it";
    popAction.setAttribute("aria-label", fixed ? `Undo, go back to ${mark.original}` : `Use ${mark.replacement}`);
    const wasPinned = Boolean(open && open.el === el && open.pinned);
    open = { el, mark, pinned: pinned || wasPinned };
    el.setAttribute("aria-expanded", "true");
    pop.hidden = false;
    place(el);
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onKey, true);
  }

  function place(el) {
    const boxRect = box.getBoundingClientRect();
    const rects = el.getClientRects();
    const rect = rects.length ? rects[rects.length - 1] : el.getBoundingClientRect();
    const width = pop.offsetWidth;
    const left = Math.max(6, Math.min(rect.left - boxRect.left - 4, box.clientWidth - width - 6));
    pop.style.left = `${left}px`;
    pop.style.top = `${rect.bottom - boxRect.top + 4}px`;
  }

  function closePop() {
    clearTimeout(hoverTimer);
    clearTimeout(closeTimer);
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onKey, true);
    if (!open) return;
    open.el.setAttribute("aria-expanded", "false");
    open = null;
    pop.hidden = true;
  }

  function scheduleClose() {
    if (!open || open.pinned) return;
    clearTimeout(closeTimer);
    closeTimer = setTimeout(closePop, 250);
  }

  function onOutside(event) {
    if (!open) return;
    const path = event.composedPath();
    if (!path.includes(pop) && !path.includes(open.el)) closePop();
  }

  function onKey(event) {
    if (event.key !== "Escape" || !open) return;
    const el = open.el;
    closePop();
    el.focus({ preventScroll: true });
  }

  textEl.addEventListener("click", (event) => {
    const el = event.target.closest(".fix");
    if (!el) return;
    if (open && open.el === el && open.pinned) closePop();
    else showPop(el, true);
  });
  textEl.addEventListener("keydown", (event) => {
    const el = event.target.closest(".fix");
    if (!el || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    showPop(el, true);
    popAction.focus({ preventScroll: true });
  });
  // A mouse shows the popover on hover; touch and keyboard use the click above.
  textEl.addEventListener("pointerover", (event) => {
    if (event.pointerType !== "mouse") return;
    const el = event.target.closest(".fix");
    if (!el) return;
    clearTimeout(closeTimer);
    if ((open && open.el === el) || (open && open.pinned)) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => showPop(el, false), 120);
  });
  textEl.addEventListener("pointerout", (event) => {
    if (event.pointerType !== "mouse" || !event.target.closest(".fix")) return;
    clearTimeout(hoverTimer);
    scheduleClose();
  });
  pop.addEventListener("pointerover", () => clearTimeout(closeTimer));
  pop.addEventListener("pointerout", (event) => {
    if (event.pointerType === "mouse") scheduleClose();
  });

  popAction.addEventListener("click", () => {
    if (!open) return;
    const { el, mark } = open;
    const id = Number(mark.id);
    const undo = isFixed(el);
    if (undo) rejected.add(id);
    else rejected.delete(id);
    // Show the change at once; the rerun then redraws the text from Python.
    el.textContent = undo ? mark.original : mark.replacement;
    el.classList.toggle("is-undone", undo);
    closePop();
    el.focus({ preventScroll: true });
    live.textContent = undo ? `Back to ${mark.original}.` : `Changed to ${mark.replacement}.`;
    inst.setStateValue("rejected", [...rejected].sort((a, b) => a - b));
  });

  // ---------------------------------------------------------------- editor
  function autosize() {
    editor.style.height = "auto";
    editor.style.height = `${editor.scrollHeight + 2}px`;
  }

  editor.addEventListener("input", autosize);
  editor.addEventListener("change", () => {
    inst.setStateValue("draft", { token: editToken, text: editor.value });
  });

  revertButton.addEventListener("click", () => inst.setTriggerValue("revert", Date.now()));

  // ------------------------------------------------------ copy and download
  function currentText() {
    return data.editing ? editor.value : String(data.text || "");
  }

  copyButton.addEventListener("click", async () => {
    const text = currentText();
    let copied = false;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch (_) {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        copied = document.execCommand("copy");
      } catch (_) {
        copied = false;
      }
      area.remove();
    }
    copyButton.classList.toggle("is-done", copied);
    copyButton.title = copied ? "Copied" : "Copy failed";
    live.textContent = copied ? "Text copied." : "The text could not be copied.";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => {
      copyButton.classList.remove("is-done");
      copyButton.title = "Copy text";
    }, 1600);
  });

  downloadButton.addEventListener("click", () => {
    const blob = new Blob([currentText()], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = data.fileName || "ingredients.txt";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  });

  // ---------------------------------------------------------------- update
  inst.closePop = closePop;
  inst.update = (next) => {
    data = next;
    rejected = new Set((Array.isArray(next.rejected) ? next.rejected : []).map(Number));
    const editing = Boolean(next.editing);
    box.classList.toggle("is-editing", editing);
    textEl.hidden = editing;
    editor.hidden = !editing;
    hint.hidden = !editing;
    foot.hidden = editing || !next.edited;
    if (editing) {
      closePop();
      if (next.editToken !== editToken) {
        editToken = next.editToken;
        editor.value = String(next.editText || "");
      }
      autosize();
      signature = null;
      return;
    }
    const nextSignature = JSON.stringify([next.text, next.marks]);
    if (nextSignature !== signature) {
      signature = nextSignature;
      renderText();
    }
  };
  return inst;
}
