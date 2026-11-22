// Ingredient list with an instant search. Everything runs in the browser.
//
// Data from Python:
//   items: [{ text, marks, strong, allergens }]   ingredients in label order
//   notes: [{ text, marks, strong, allergens }]   statements such as "Contains: milk"
// "marks" are character ranges of words read with low confidence, "strong" the
// ranges of allergen words (bold, as on real labels) and "allergens" the group
// names shown as tags. Searching matches the text and the group names.

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

// Text with <mark> (low confidence) and <strong> (allergen) ranges plus allergen
// tags, built without innerHTML.
function fill(target, entry) {
  const text = String(entry.text || "");
  const flags = new Uint8Array(text.length);
  const flag = (ranges, bit) => {
    for (const range of Array.isArray(ranges) ? ranges : []) {
      if (!Array.isArray(range)) continue;
      const end = Math.min(text.length, Number(range[1]) || 0);
      for (let i = Math.max(0, Number(range[0]) || 0); i < end; i++) flags[i] |= bit;
    }
  };
  flag(entry.marks, LOW);
  flag(entry.strong, ALLERGEN);
  for (let i = 0; i < text.length; ) {
    let j = i + 1;
    while (j < text.length && flags[j] === flags[i]) j++;
    let node = document.createTextNode(text.slice(i, j));
    if (flags[i] & ALLERGEN) {
      const strong = document.createElement("strong");
      strong.append(node);
      node = strong;
    }
    if (flags[i] & LOW) {
      const mark = document.createElement("mark");
      mark.title = "Read with low confidence. Check it on the label.";
      mark.append(node);
      node = mark;
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
      tag.className = "tag";
      tag.textContent = group;
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

  function render(items, notes) {
    rows = items.map((item) => {
      const li = document.createElement("li");
      const txt = document.createElement("span");
      txt.className = "txt";
      fill(txt, item);
      li.append(txt);
      return { el: li, key: searchKey(item) };
    });
    list.replaceChildren(...rows.map((row) => row.el));
    list.hidden = rows.length === 0;
    noteRows = notes.map((note) => {
      const li = document.createElement("li");
      fill(li, note);
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
      const next = JSON.stringify([items, notes]);
      if (next !== signature) {
        signature = next;
        render(items, notes);
      }
      search();
    },
  };
}
