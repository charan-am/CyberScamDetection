# Third-Party Notices

This project's own source code is licensed under the MIT License (see `LICENSE`).
That license does **not** cover third-party data or software included in or used by
this project.

## UCI SMS Spam Collection

- **Dataset:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository
- **Authors / citation:** Tiago A. Almeida, José María Gómez Hidalgo, and Akebo Yamakami,
  "Contributions to the Study of SMS Spam Filtering: New Collection and Results",
  Proceedings of the 11th ACM Symposium on Document Engineering (DocEng 2011).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), as published
  with the dataset.

The dataset is redistributed here with attribution under the CC BY 4.0 terms
published with it by the UCI Machine Learning Repository; CC BY 4.0 permits reuse,
including commercial reuse, provided attribution is given. The dataset's bundled
`readme` also asks users to cite the authors' paper and to notify them of use.
The corpus is committed at `data/raw/SMSSpamCollection` with the archive's own
`data/raw/readme`.

- **Download:** `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- **SHA-256:** `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`
- **Rows:** 5,574 (4,827 ham, 747 spam)

Full provenance and preparation notes: `data/README.md` and `docs/dataset.md`.

## Python dependencies

Runtime and development dependencies are listed with pinned versions in
`requirements.txt` and `requirements-dev.txt`. Each dependency retains its own
license; this project does not relicense them.

| Dependency | Licence |
|---|---|
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause |
| joblib | BSD-3-Clause |
| streamlit | Apache-2.0 |
| pytest | MIT |
| pyarrow | Apache-2.0 |

Transitive dependencies keep their own licences as recorded in their distributions.

