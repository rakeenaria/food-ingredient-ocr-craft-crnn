"""About: how the reader works, how accurate it is, and its limits."""

import streamlit as st

from ui import style

style.inject()
style.masthead("About", "A thesis project turned into a tool for reading ingredient lists from photos of food labels.")

style.section("What it does")
st.html(
    "<p>Ingredient lists are printed small and dense, often on glossy, curved packaging. "
    "Ingredient Facts turns a photo of that list into plain text you can copy or save, "
    "and into a numbered list of ingredients you can search.</p>"
)

style.section("How it works")
st.html(
    '<ol class="ir-list">'
    "<li><b>Find the text.</b> CRAFT, a character-level text detector, locates every word on the photo.</li>"
    "<li><b>Straighten each word.</b> Each word is cut out and perspective-corrected.</li>"
    "<li><b>Read each word.</b> A CRNN (TPS-ResNet-BiLSTM-Attention), fine-tuned on 6,081 word images "
    "from food labels, recognizes the characters.</li>"
    "<li><b>Put it back together.</b> Words are reassembled in reading order, left to right and top to bottom.</li>"
    "<li><b>Flag doubtful words.</b> Every word gets a confidence score, and words under 80% get a dotted underline. "
    "On the thesis test set this flagged 6% of the words and caught 12 of the 14 words with misread "
    "letters or digits.</li>"
    "<li><b>Split the list.</b> Commas outside brackets separate the ingredients. Statements such as "
    "“Contains: milk” are listed apart.</li>"
    "<li><b>Spot allergens.</b> Words for nine common allergen groups are marked: milk, egg, peanut, tree nuts, "
    "soy, wheat and other gluten cereals, fish, shellfish and sesame. “May contain” statements are kept "
    "apart from the ingredients.</li>"
    "<li><b>Fix misreadings.</b> A word that is not in an ingredient vocabulary but is a letter or two away from "
    "a word that is gets corrected, such as “Vilater” to “Water”. Corrections are marked in yellow, and you can "
    "undo each one.</li>"
    "</ol>"
)

style.section("Accuracy")
style.facts(
    [
        ("Word accuracy (test set)", "94.693%"),
        ("Character error rate", "0.015"),
        ("Word error rate", "0.053"),
        ("Word accuracy before fine-tuning", "90.381%"),
        ("Tesseract on the same test set", "70.813%"),
        ("Word accuracy on 5 real product photos", "92.248%"),
    ]
)
style.note(
    "Results from the thesis behind this app. The test set is 10% of 6,081 curated word images "
    "from the Kaggle Ingredients Image from Food Label dataset (80:10:10 split)."
)

style.section("Limits")
st.html(
    '<ul class="ir-list">'
    "<li>Printed English ingredient lists. Handwriting and other languages are not supported.</li>"
    "<li>One still photo at a time. There is no live video scanning.</li>"
    "<li>Small, blurry, glossy or curved text lowers accuracy. Crop to the list and use good light.</li>"
    "<li>Allergens are found by matching words. Unusual names and misread words can be missed, "
    "and an automatic correction can be wrong.</li>"
    "<li>OCR can be wrong. Always check the physical label, especially if you have an allergy.</li>"
    "</ul>"
)

style.section("Privacy")
st.html(
    "<p>Your photo is sent to the server only to read it. It stays in memory for your session "
    "and is never written to disk or logged. Leaving the page discards it.</p>"
)

style.section("Author")
st.html(
    "<p>Built by <b>Rakeen Aria Alireza</b> as an undergraduate thesis project: "
    "<i>Pengembangan Sistem OCR Berbasis CRAFT dan CRNN untuk Pengenalan Teks Daftar Bahan "
    "pada Label Makanan</i>.</p>"
    '<p><a href="https://github.com/rakeenaria">GitHub: rakeenaria</a> · '
    '<a href="https://github.com/rakeenaria/food-ingredient-ocr-craft-crnn">Source code</a></p>'
)

style.section("Credits")
st.html(
    "<p>CRAFT text detector and the deep-text-recognition-benchmark framework by Clova AI Research, NAVER. "
    "Built with PyTorch, OpenCV and Streamlit.</p>"
)
