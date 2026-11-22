// Ingredient list with an instant search. Everything runs in the browser.
//
// Data from Python:
//   items: [{ text, low, fixed, chips, allergens }]          ingredients in label order
//   notes: [{ text, low, fixed, chips, allergens, maybe }]   statements such as "Contains: milk"
//   palette: { group: { fill, ink, line } }                  allergen colors
// "low" are character ranges of words read with low confidence (yellow dotted
// underline), "fixed" ranges of words corrected automatically ([start, end,
// word as read], green dotted underline) and "chips" ranges of allergen words
// ([start, end, group]), shown in the colors of their group as in the label
// text. "allergens" are the group names, shown as tags after the text, so a
// word such as "whey" says which group it belongs to. A "may contain"
// statement ("maybe") has outlined words and tags.
// Searching matches the text and the group names.

const instances = new WeakMap();

export default function ({ parentElement, data }) {
  let inst = instances.get(parentElement);
  if (!inst) {
    inst = createInstance(parentElement);
    instances.set(parentElement, inst);
  }
  inst.update(data || {});
}

function normalize(text) {
  return String(text || "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

// [start, end, extra] ranges from Python, in text order, without overlaps or
// ranges outside the text.
function ranges(lists, length) {
  const all = [];
  for (const [list, kind] of lists) {
    for (const range of Array.isArray(list) ? list : []) {
      if (!Array.isArray(range)) continue;
      all.push({ start: Number(range[0]), end: Number(range[1]), extra: String(range[2] || ""), kind });
    }
  }
  const kept = [];
  let cursor = 0;
  for (const range of all.filter((r) => r.end > r.start).sort((a, b) => a.start - b.start)) {
    if (range.start < cursor || range.start < 0 || range.end > length) continue;
    kept.push(range);
    cursor = range.end;
  }
  return kept;
}

function paint(el, colors) {
  if (!colors) return;
  el.style.setProperty("--tag", colors.fill);
  el.style.setProperty("--tag-ink", colors.ink);
  el.style.setProperty("--tag-line", colors.line);
}

// A word to check or a corrected word; it holds the pieces of that word.
function markElement(mark) {
  const span = document.createElement("span");
  if (mark.kind === "fixed") {
    span.className = "fixed";
    span.title = `Corrected automatically. Read as “${mark.extra}”.`;
  } else {
    span.className = "low";
    span.title = "Read with low confidence. Check it on the label.";
  }
  return span;
}

// An allergen word, or the part of one that falls inside or outside a mark
// (as in label_text.js).
function chipElement(piece, group, maybe, isStart, isEnd, palette, titled) {
  const span = document.createElement("span");
  span.className = `chip${maybe ? " is-maybe" : ""}${isStart ? " is-start" : ""}${isEnd ? " is-end" : ""}`;
  paint(span, palette[group]);
  if (titled) span.title = maybe ? `May contain: ${group}` : `Allergen: ${group}`;
  span.textContent = piece;
  return span;
}

// The text with its dotted words and colored allergen words, then the group
// tags, built without innerHTML.
function fill(target, entry, palette) {
  const text = String(entry.text || "");
  const maybe = Boolean(entry.maybe);
  const marks = ranges(
    [
      [entry.low, "low"],
      [entry.fixed, "fixed"],
    ],
    text.length,
  );
  const chips = ranges([[entry.chips, "chip"]], text.length);
  // Cut the text wherever a mark or an allergen word starts or ends, then give
  // each mark (one word) its pieces.
  const cuts = new Set([0, text.length]);
  for (const range of [...marks, ...chips]) {
    cuts.add(range.start);
    cuts.add(range.end);
  }
  const points = [...cuts].sort((a, b) => a - b);
  let m = 0;
  let c = 0;
  let markEl = null;
  let markOf = null;
  for (let k = 0; k + 1 < points.length; k++) {
    const start = points[k];
    const end = points[k + 1];
    while (m < marks.length && marks[m].end <= start) m++;
    while (c < chips.length && chips[c].end <= start) c++;
    const mark = m < marks.length && marks[m].start <= start ? marks[m] : null;
    const chip = c < chips.length && chips[c].start <= start ? chips[c] : null;
    const piece = text.slice(start, end);
    const node = chip
      ? chipElement(piece, chip.extra, maybe, start === chip.start, end === chip.end, palette, !mark)
      : document.createTextNode(piece);
    if (!mark) {
      markEl = markOf = null;
      target.append(node);
      continue;
    }
    if (markOf !== mark) {
      markOf = mark;
      markEl = markElement(mark);
      target.append(markEl);
    }
    markEl.append(node);
  }
  const groups = Array.isArray(entry.allergens) ? entry.allergens : [];
  if (groups.length) {
    const box = document.createElement("span");
    box.className = "tags";
    for (const group of groups) {
      const tag = document.createElement("span");
      tag.className = maybe ? "tag is-maybe" : "tag";
      tag.textContent = group;
      paint(tag, palette[group]);
      box.append(tag);
    }
    target.append(box);
  }
}

function searchKey(entry) {
  const groups = Array.isArray(entry.allergens) ? entry.allergens : [];
  return normalize(`${entry.text || ""} ${groups.join(" ")}`);
}

function plural(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}

function createInstance(root) {
  const input = root.querySelector("input");
  const status = root.querySelector(".status");
  const list = root.querySelector(".items");
  const notesBox = root.querySelector(".notes");
  const noteList = root.querySelector(".note-list");
  let rows = [];
  let noteRows = [];
  let signature = null;

  function render(items, notes, palette) {
    rows = items.map((item) => {
      const li = document.createElement("li");
      const txt = document.createElement("span");
      txt.className = "txt";
      fill(txt, item, palette);
      li.append(txt);
      return { el: li, key: searchKey(item) };
    });
    list.replaceChildren(...rows.map((row) => row.el));
    list.hidden = rows.length === 0;
    noteRows = notes.map((note) => {
      const li = document.createElement("li");
      fill(li, note, palette);
      return { el: li, key: searchKey(note) };
    });
    noteList.replaceChildren(...noteRows.map((row) => row.el));
    notesBox.hidden = noteRows.length === 0;
  }

  function mark(rowsToMark, query) {
    let hits = 0;
    for (const row of rowsToMark) {
      const match = query !== "" && row.key.includes(query);
      row.el.classList.toggle("is-match", match);
      row.el.classList.toggle("is-dim", query !== "" && !match);
      if (match) hits += 1;
    }
    return hits;
  }

  function search() {
    const query = normalize(input.value);
    const hits = mark(rows, query);
    const noteHits = mark(noteRows, query);
    if (!query) {
      status.textContent = "";
      return;
    }
    const shown = `“${input.value.trim()}”`;
    if (hits === 0 && noteHits === 0) {
      status.textContent = `Nothing matches ${shown}. OCR can misread words, so check the label too.`;
      return;
    }
    const ingredients = plural(hits, "ingredient", "ingredients");
    const notes = plural(noteHits, "label note", "label notes");
    if (hits && noteHits) {
      status.textContent = `${ingredients} and ${notes} match ${shown}.`;
    } else if (hits) {
      status.textContent = `${ingredients} ${hits === 1 ? "matches" : "match"} ${shown}.`;
    } else {
      status.textContent = `No ingredient matches ${shown}, but ${notes} ${noteHits === 1 ? "does" : "do"}.`;
    }
  }

  input.addEventListener("input", search);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && input.value) {
      input.value = "";
      search();
    }
  });

  return {
    update(data) {
      const items = Array.isArray(data.items) ? data.items : [];
      const notes = Array.isArray(data.notes) ? data.notes : [];
      const palette = data.palette && typeof data.palette === "object" ? data.palette : {};
      const next = JSON.stringify([items, notes, palette]);
      if (next !== signature) {
        signature = next;
        render(items, notes, palette);
      }
      search();
    },
  };
}
