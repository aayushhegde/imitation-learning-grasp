---
name: ieee-citation
description: Format a research paper's details as an IEEE-style BibTeX entry and append it to references.bib. Use when the user gives paper details (title, authors, venue, year, etc.) and wants it added as a citation/reference.
---

# ieee-citation

Turn a paper's details into a BibTeX entry formatted for IEEE style, and append it to
`references.bib` in the project root (create the file if it doesn't exist).

## 1. Collect the fields

Ask for whatever the user didn't already provide:

- **Required**: authors, title, year, venue (journal or conference name)
- **Optional**: volume, number (issue), pages, month, DOI

Determine the venue type from what the user says: a journal name → `@article`; a
conference/workshop/symposium name → `@inproceedings`.

## 2. Build the citation key

`<firstAuthorLastName><year><firstTitleWord>`, all lowercase, no spaces/punctuation.
Skip a leading article ("A", "An", "The") when picking the title word.
Example: "Ken Goldberg, ... , 2023, 'Grasping in Cluttered Scenes'" → `goldberg2023grasping`.
Example: "Ross et al., 2011, 'A Reduction of Imitation Learning...'" → `ross2011reduction`.

## 3. Format the entry

Author field: `Last, First and Last, First and ...` (BibTeX's expected format so any
`.bst` — including IEEE's `IEEEtran.bst` — renders it correctly).

Title field: wrap in double braces, `{{Title Here}}`, so BibTeX preserves the original
capitalization (IEEE style keeps proper nouns/acronyms capitalized as given).

For an `@article`:

```
@article{goldberg2023grasping,
  author  = {Goldberg, Ken and Doe, Jane},
  title   = {{Grasping in Cluttered Scenes}},
  journal = {IEEE Transactions on Robotics},
  year    = {2023},
  volume  = {39},
  number  = {2},
  pages   = {123--134},
  doi     = {10.1109/TRO.2023.1234567}
}
```

For an `@inproceedings`, use `booktitle` instead of `journal`/`volume`/`number`, e.g.
`booktitle = {Proc. IEEE Int. Conf. on Robotics and Automation (ICRA)}`.

Omit any field the user didn't supply — don't invent placeholder values.

## 4. Append to references.bib

- If `references.bib` doesn't exist, create it.
- Add the new entry at the end of the file, separated from existing entries by a blank line.
- Don't reformat or reorder existing entries — only add the new one.

## 5. Confirm

Show the user the exact entry that was added and its citation key (so they know what to
use with `\cite{...}` in their LaTeX).
