// Ingredient list with an instant search. Everything runs in the browser.
//
// Data from Python:
//   items: [{ text, marks: [[start, end], ...] }]   ingredients in label order
//   notes: [{ text, marks }]                         statements such as "Contains: milk"
// "marks" are character ranges of words read with low confidence.

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

// Text with <mark> around low-confidence ranges, built without innerHTML.
function fill(target, entry) {
  const text = String(entry.text || "");
  const marks = (Array.isArray(entry.marks) ? entry.marks : [])
    .filter((m) => Array.isArray(m) && m[0] < m[1])
    .sort((a, b) => a[0] - b[0]);
  let cursor = 0;
  for (const [start, end] of marks) {
    if (start < cursor) continue;
    if (start > cursor) target.append(text.slice(cursor, start));
    const mark = document.createElement("mark");
    mark.textContent = text.slice(start, end);
    mark.title = "Read with low confidence. Check it on the label.";
    target.append(mark);
    cursor = end;
  }
  if (cursor < text.length) target.append(text.slice(cursor));
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
      return { el: li, key: normalize(item.text) };
    });
    list.replaceChildren(...rows.map((row) => row.el));
    list.hidden = rows.length === 0;
    noteRows = notes.map((note) => {
      const li = document.createElement("li");
      fill(li, note);
      return { el: li, key: normalize(note.text) };
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
