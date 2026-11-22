// Ingredient list with an instant search. Everything runs in the browser.
//
// Data from Python:
//   items: [{ text, low, fixed, strong, allergens }]          ingredients in label order
//   notes: [{ text, low, fixed, strong, allergens, maybe }]   statements such as "Contains: milk"
//   palette: { group: { fill, ink, line } }            allergen tag colors
// "low" are character ranges of words read with low confidence (yellow dotted
// underline), "fixed" ranges of words corrected automatically ([start, end,
// word as read], green dotted underline), "strong" the ranges of allergen
// words (bold, as on real labels) and "allergens" the group names shown as
// colored tags (outlined for a "may contain" statement, "maybe").
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

const LOW = 1;
const ALLERGEN = 2;
const FIXED = 4;

// Text with dotted (low confidence, corrected) and <strong> (allergen) ranges
// plus colored allergen tags, built without innerHTML.
function fill(target, entry, palette) {
  const text = String(entry.text || "");
  const flags = new Uint8Array(text.length);
  const readAs = new Array(text.length).fill("");
  const flag = (ranges, bit) => {
    for (const range of Array.isArray(ranges) ? ranges : []) {
      if (!Array.isArray(range)) continue;
      const end = Math.min(text.length, Number(range[1]) || 0);
      for (let i = Math.max(0, Number(range[0]) || 0); i < end; i++) {
        flags[i] |= bit;
        if (bit === FIXED) readAs[i] = String(range[2] || "");
      }
    }
  };
  flag(entry.low, LOW);
  flag(entry.strong, ALLERGEN);
  flag(entry.fixed, FIXED);
  for (let i = 0; i < text.length; ) {
    let j = i + 1;
    while (j < text.length && flags[j] === flags[i] && readAs[j] === readAs[i]) j++;
    let node = document.createTextNode(text.slice(i, j));
    if (flags[i] & ALLERGEN) {
      const strong = document.createElement("strong");
      strong.append(node);
      node = strong;
    }
    if (flags[i] & FIXED) {
      const fixed = document.createElement("span");
      fixed.className = "fixed";
      fixed.title = `Corrected automatically. Read as “${readAs[i]}”.`;
      fixed.append(node);
      node = fixed;
    } else if (flags[i] & LOW) {
      const low = document.createElement("span");
      low.className = "low";
      low.title = "Read with low confidence. Check it on the label.";
      low.append(node);
      node = low;
    }
    target.append(node);
    i = j;
  }
  const groups = Array.isArray(entry.allergens) ? entry.allergens : [];
  if (groups.length) {
    const box = document.createElement("span");
    box.className = "tags";
    for (const group of groups) {
      const tag = document.createElement("span");
      tag.className = entry.maybe ? "tag is-maybe" : "tag";
      tag.textContent = group;
      const colors = palette && palette[group];
      if (colors) {
        tag.style.setProperty("--tag", colors.fill);
        tag.style.setProperty("--tag-ink", colors.ink);
        tag.style.setProperty("--tag-line", colors.line);
      }
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
