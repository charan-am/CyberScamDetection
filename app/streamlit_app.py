"""Streamlit interface for the AI-Based Cyber Scam Detection and Awareness System.

Run with:  streamlit run app\\streamlit_app.py

Design rules enforced here:
* the app never trains a model; it only loads the checksummed artifact;
* the model score is shown as uncalibrated output, never as a confidence;
* model explanation, rule-based indicators and awareness guidance are three
  visually separate layers and are never merged into one claim;
* submitted messages are processed in memory and never stored.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app/streamlit_app.py` from a checkout without installing.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from scamdetector import __version__, awareness, config, present
from scamdetector.explain import detect_signals
from scamdetector.predict import (
    ArtifactError,
    MessageValidationError,
    is_probably_non_english,
    load_model,
    predict_message,
)
from scamdetector import evaluate

EXAMPLE_MESSAGE = (
    "Congratulations! Your mobile number won Rs 25,00,000 in the KBC lucky draw. "
    "To claim the prize, pay Rs 6,500 processing fee and send your bank details "
    "and OTP on WhatsApp within 24 hours."
)

SIGNAL_TO_CATEGORY = {
    "otp": "otp",
    "url": "phishing",
    "short_url": "phishing",
    "email": "phishing",
    "prize": "lottery",
    "job": "fake_jobs",
    "investment": "investment",
    "payment": "upi",
    "amount": "upi",
    "urgency": "banking",
    "phone": "banking",
}

CSS = """
<style>
:root { color-scheme: dark; }
.stApp {
  background: radial-gradient(circle at 15% 0%, #10243b 0%, #0a1420 45%, #070d15 100%);
}
.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.hero {
  border: 1px solid #1d3a55;
  background: linear-gradient(135deg, rgba(15,42,66,0.95), rgba(9,22,36,0.95));
  border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.4rem;
  box-shadow: 0 18px 40px rgba(0,0,0,0.35);
}
.hero h1 { font-size: 1.9rem; margin: 0 0 .35rem 0; color: #eaf6ff; }
.hero p { margin: 0; color: #9fc2da; font-size: 1rem; }
.badges { margin-top: .9rem; }
.badge {
  display: inline-block; font-size: .76rem; letter-spacing: .04em;
  text-transform: uppercase; color: #8fe3ff; background: rgba(23,162,184,0.12);
  border: 1px solid rgba(89,201,232,0.45); border-radius: 999px;
  padding: .28rem .7rem; margin-right: .45rem; margin-top: .3rem;
}
.result {
  border-radius: 14px; padding: 1.1rem 1.3rem; margin: .4rem 0 1rem 0;
  border: 1px solid #23405a; background: rgba(13,30,46,0.85);
}
.result.spam { border-left: 6px solid #ff6b6b; }
.result.ham { border-left: 6px solid #35d07f; }
.result h3 { margin: 0 0 .3rem 0; font-size: 1.25rem; }
.small { color: #9db4c7; font-size: .88rem; }
.layer-title {
  font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
  color: #7fd4ea; margin-bottom: .2rem;
}
div[data-testid="stMetric"] {
  background: rgba(14,32,49,0.85); border: 1px solid #1d3a55;
  border-radius: 12px; padding: .7rem .9rem;
}
div[data-testid="stMetricLabel"] p { color: #9db4c7; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] {
  background: rgba(14,32,49,0.7); border: 1px solid #1d3a55;
  border-radius: 10px 10px 0 0; padding: .5rem 1rem;
}
.stTabs [aria-selected="true"] { background: rgba(23,162,184,0.18); }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid #2a6f8f; font-weight: 600;
}
.quote { border-left: 4px solid #59c9e8; background: rgba(13,30,46,0.7); padding: .8rem 1rem; border-radius: 0 10px 10px 0; }
</style>
"""


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _hero() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Cyber Scam Detection &amp; Awareness</h1>
  <p>An explainable machine-learning check for English SMS, with verified Indian
  cyber-safety guidance and honest performance reporting.</p>
  <div class="badges">
    <span class="badge">TF-IDF + Logistic Regression</span>
    <span class="badge">English SMS only</span>
    <span class="badge">Runs locally</span>
    <span class="badge">No message storage</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _artifact() -> dict:
    """Load and cache the checksummed model artifact for this process."""
    return load_model()


@st.cache_data(show_spinner=False)
def _metrics() -> dict:
    return evaluate.load_metrics()


def _guidance_for(signal_codes: list[str], language: str) -> list[awareness.Category]:
    wanted: list[str] = []
    for code in signal_codes:
        category_id = SIGNAL_TO_CATEGORY.get(code)
        if category_id and category_id not in wanted:
            wanted.append(category_id)
    if not wanted:
        wanted = ["phishing", "otp"]
    content = awareness.load_awareness(language)
    by_id = {category.id: category for category in content.categories}
    return [by_id[category_id] for category_id in wanted if category_id in by_id][:2]


def _render_result(result, message: str) -> None:
    css_class = "spam" if result.flagged else "ham"
    st.markdown(
        f"""
<div class="result {css_class}">
  <div class="layer-title">Layer 1 — model prediction</div>
  <h3>{present.label_text(result.label)}</h3>
  <div class="small">{present.score_caption(result.raw_score, result.threshold)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if result.message_length and is_probably_non_english(message):
        st.info(
            "This message does not look like English. The model was trained on "
            "English SMS only, so treat this prediction as unreliable.",
            icon="⚠️",
        )

    st.markdown("**Why the model decided this**")
    st.caption(present.explanation_intro())
    left, right = st.columns(2)
    with left:
        st.markdown("Words pushing towards **spam**")
        rows = present.contribution_rows(result.top_positive)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("No strongly spam-like words were found in this message.")
    with right:
        st.markdown("Words pushing towards **ham**")
        rows = present.contribution_rows(result.top_negative)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("No strongly ham-like words were found in this message.")

    st.divider()

    st.markdown(
        '<div class="layer-title">Layer 2 — rule-based message characteristics</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These patterns are found by fixed rules in the text. They are not part "
        "of the model's decision and they do not prove the message is a scam."
    )
    signals = detect_signals(message)
    if signals:
        st.dataframe(
            present.signal_rows(signals), hide_index=True, width="stretch"
        )
    else:
        st.caption(
            "No common scam patterns (links, OTP requests, prize claims, urgency) "
            "were detected."
        )

    st.divider()

    st.markdown(
        '<div class="layer-title">Layer 3 — verified awareness guidance</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Curated safety guidance from this project's content files, independent "
        "of the model. It is general advice, not a judgement about this message."
    )
    for category in _guidance_for([signal.code for signal in signals], "en"):
        with st.expander(f"{category.title} — what to check", expanded=True):
            st.write(category.summary)
            st.markdown("**Warning signs**")
            for item in category.warning_signs:
                st.markdown(f"- {item}")
            st.markdown("**What to do**")
            for item in category.prevention:
                st.markdown(f"- {item}")

    st.caption(present.disclaimer_text())


def _detection_tab() -> None:
    st.subheader("Check a message")
    st.caption(
        "Paste an SMS you received. Do not paste real OTPs, PINs, card numbers "
        "or account numbers — the tool does not need them."
    )

    if "message_input" not in st.session_state:
        st.session_state.message_input = ""

    message = st.text_area(
        "Message text",
        key="message_input",
        height=170,
        max_chars=config.MAX_MESSAGE_CHARS,
        placeholder="Paste the SMS here…",
        label_visibility="collapsed",
    )

    left, right, _ = st.columns([1, 1, 2])
    analyse = left.button(
        "Analyse message", type="primary", width="stretch"
    )
    clear = right.button("Clear", width="stretch")

    if clear:
        st.session_state.message_input = ""
        st.session_state.pop("result", None)
        st.session_state.pop("analysed_text", None)
        st.rerun()

    if analyse:
        try:
            artifact = _artifact()
            st.session_state.result = predict_message(message, artifact=artifact)
            st.session_state.analysed_text = message
        except MessageValidationError as exc:
            st.session_state.pop("result", None)
            st.warning(str(exc), icon="✏️")
        except ArtifactError as exc:
            st.session_state.pop("result", None)
            st.error(f"The model could not be loaded safely. {exc}", icon="🛑")
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.session_state.pop("result", None)
            st.error(
                "Something went wrong while analysing the message. "
                f"Technical detail: {type(exc).__name__}.",
                icon="🛑",
            )

    result = st.session_state.get("result")
    if result is not None:
        _render_result(result, st.session_state.get("analysed_text", ""))
    else:
        with st.expander("Try an example message"):
            st.code(EXAMPLE_MESSAGE, language=None)
            st.caption("Copy it into the box above, then choose Analyse message.")


def _awareness_tab() -> None:
    st.subheader("Cyber awareness")
    label_to_code = {"English": "en", "ಕನ್ನಡ (Kannada)": "kn"}
    choice = st.radio(
        "Content language",
        list(label_to_code),
        horizontal=True,
        help=(
            "The classifier itself works on English SMS. This switch changes the "
            "guidance language only."
        ),
    )
    language = label_to_code[choice]

    try:
        content = awareness.load_awareness(language)
    except awareness.ContentError as exc:
        st.error(f"The awareness content could not be loaded: {exc}")
        return

    if not content.reviewed:
        st.warning(
            content.review_note
            or "This translation has not yet been independently reviewed.",
            icon="🔎",
        )
    else:
        st.caption(f"Content last updated {content.last_updated}.")

    titles = [category.title for category in content.categories]
    selected_title = st.selectbox("Scam category", titles)
    category = next(c for c in content.categories if c.title == selected_title)

    st.markdown(f"### {category.title}")
    st.write(category.summary)

    st.markdown(
        f'<div class="quote"><strong>Example message</strong><br>{category.example}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Warning signs**")
        for item in category.warning_signs:
            st.markdown(f"- {item}")
    with col_b:
        st.markdown("**How to protect yourself**")
        for item in category.prevention:
            st.markdown(f"- {item}")

    st.markdown("**If it already happened — checklist**")
    for index, item in enumerate(category.checklist):
        st.checkbox(
            item, key=f"check-{language}-{category.id}-{index}"
        )

    st.divider()
    _reporting_block()


def _reporting_block() -> None:
    st.markdown("### Where to report in India")
    try:
        info = awareness.load_reporting()
    except awareness.ContentError as exc:
        st.error(f"Reporting information could not be loaded: {exc}")
        return

    st.caption(
        f"Official channels, last verified {info.last_verified}. "
        "Use the helpline immediately if money has already left your account."
    )
    for channel in info.channels:
        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{channel.name}**")
                st.caption(channel.operator)
                st.write(channel.purpose)
                if channel.note:
                    st.caption(channel.note)
            with cols[1]:
                st.link_button("Open portal", channel.url, width="stretch")
                if channel.phone:
                    st.markdown(f"**Helpline: {channel.phone}**")
    st.caption(info.disclaimer)


def _performance_tab() -> None:
    st.subheader("Model performance")
    try:
        metrics = _metrics()
    except Exception as exc:
        st.info(
            "No metrics file is available yet. Train the model with "
            "`python scripts\\train_model.py` to generate one."
        )
        st.caption(f"Detail: {type(exc).__name__}")
        return

    test = metrics["test"]
    st.caption(
        "All numbers below come from a single evaluation on the held-out test "
        "split, which was not used for training or for choosing the threshold."
    )

    cols = st.columns(4)
    cols[0].metric("F1 (spam)", present.format_metric(test["f1"]))
    cols[1].metric("Precision", present.format_metric(test["precision"]))
    cols[2].metric("Recall", present.format_metric(test["recall"]))
    cols[3].metric("Accuracy", present.format_metric(test["accuracy"]))

    cols = st.columns(3)
    cols[0].metric("ROC-AUC", present.format_metric(test.get("roc_auc")))
    cols[1].metric("PR-AUC", present.format_metric(test.get("pr_auc")))
    cols[2].metric("Test messages", f"{test['rows']}")

    st.markdown("### Confusion matrix (test split)")
    matrix = test["confusion_matrix"]
    st.dataframe(
        [
            {
                "Actual": "ham",
                "Predicted ham": matrix["tn"],
                "Predicted spam": matrix["fp"],
            },
            {
                "Actual": "spam",
                "Predicted ham": matrix["fn"],
                "Predicted spam": matrix["tp"],
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Support: {test['support']['ham']} ham and "
        f"{test['support']['spam']} spam messages."
    )

    st.markdown("### Comparison with baselines")
    rows = [
        {
            "Model": "TF-IDF + Logistic Regression",
            "Precision": present.format_metric(test["precision"]),
            "Recall": present.format_metric(test["recall"]),
            "F1": present.format_metric(test["f1"]),
            "PR-AUC": present.format_metric(test.get("pr_auc")),
        }
    ]
    labels = {
        "majority_class": "Majority class (always ham)",
        "keyword_rule": "Single-keyword rule",
    }
    for key, row in metrics["baselines"].items():
        rows.append(
            {
                "Model": labels.get(key, key),
                "Precision": present.format_metric(row["precision"]),
                "Recall": present.format_metric(row["recall"]),
                "F1": present.format_metric(row["f1"]),
                "PR-AUC": present.format_metric(row.get("pr_auc")),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("### Configuration")
    model = metrics["model"]
    st.dataframe(
        [
            {"Setting": "Classifier", "Value": model["type"]},
            {"Setting": "Regularisation C", "Value": str(model["C"])},
            {
                "Setting": "N-grams",
                "Value": f"{model['ngram_range'][0]}–{model['ngram_range'][1]}",
            },
            {"Setting": "Class weight", "Value": model["class_weight"]},
            {
                "Setting": "Decision threshold",
                "Value": f"{float(model['threshold']):.2f}",
            },
            {
                "Setting": "Threshold chosen on",
                "Value": model["threshold_selected_on"],
            },
            {
                "Setting": "Threshold objective",
                "Value": model["threshold_objective"],
            },
            {
                "Setting": "Dataset rows (raw)",
                "Value": str(metrics["dataset"]["raw_rows"]),
            },
            {
                "Setting": "Test evaluated once",
                "Value": str(metrics["dataset"]["test_evaluated_once"]),
            },
        ],
        hide_index=True,
        width="stretch",
    )

    st.markdown("### What these numbers do not cover")
    st.markdown(
        """
- The dataset is a 2011-era collection of English SMS. Modern scams delivered
  over WhatsApp, RCS, email or voice calls are not represented.
- Only English is supported. Kannada and other Indian-language messages are
  outside the training data, so predictions on them are unreliable.
- Spam in this dataset means unwanted or fraudulent SMS. The model does not
  distinguish between scam types and cannot confirm that a message is a crime.
- The score is uncalibrated and is shown only as raw model output.
- Performance on real, current messages can differ from these results.
"""
    )


def _about_tab() -> None:
    st.subheader("About, data and privacy")
    st.markdown(
        """
**What this tool is.** A student project that demonstrates an explainable
text classifier for English SMS, paired with Indian cyber-awareness guidance.
It is a learning and awareness aid, not a security product.

**How a prediction is produced.** The message is normalised, sensitive tokens
(links, emails, amounts, phone numbers) are replaced with placeholders, the text
is converted to TF-IDF features, and a logistic-regression model produces a raw
score. That score is compared with a threshold chosen on a validation split.
"""
    )

    try:
        artifact = _artifact()
        status = "loaded and checksum-verified"
        details = [
            {
                "Item": "Model type",
                "Value": str(artifact["model_params"].get("type", "LogisticRegression")),
            },
            {"Item": "Trained at", "Value": str(artifact["trained_at"])},
            {"Item": "scikit-learn", "Value": str(artifact["sklearn_version"])},
            {
                "Item": "Dataset SHA-256",
                "Value": artifact["dataset_hash"][:16] + "…",
            },
            {"Item": "Config hash", "Value": artifact["config_hash"][:16] + "…"},
            {
                "Item": "Threshold",
                "Value": f"{float(artifact['threshold']):.2f}",
            },
        ]
    except Exception as exc:
        status = "not loaded"
        details = [{"Item": "Load error", "Value": type(exc).__name__}]

    st.markdown(f"**Model artifact:** {status}.")
    st.dataframe(details, hide_index=True, width="stretch")

    st.markdown("### Dataset")
    st.markdown(
        """
The model was trained on the **UCI SMS Spam Collection** (Almeida, Gómez Hidalgo
and Yamakami, 2011), distributed by the UCI Machine Learning Repository under
**CC BY 4.0**. The dataset is used unchanged apart from removing duplicate
messages before splitting. It is not redistributed with the application's own
MIT-licensed code; attribution is recorded in `THIRD_PARTY_NOTICES.md` and
`docs/dataset.md`.
"""
    )
    st.link_button(
        "UCI SMS Spam Collection",
        "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    )

    st.markdown("### Privacy")
    st.write(present.privacy_note())
    st.caption(
        "The app runs entirely on the local machine. There are no external API "
        "calls for inference and no analytics or tracking."
    )

    st.markdown("### Project details")
    st.dataframe(
        [
            {"Item": "Application version", "Value": __version__},
            {"Item": "Python", "Value": "3.12.16"},
            {"Item": "Licence (original code)", "Value": "MIT"},
            {"Item": "Dataset licence", "Value": "CC BY 4.0 (UCI)"},
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(present.disclaimer_text())


def main() -> None:
    st.set_page_config(
        page_title="Cyber Scam Detection & Awareness",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()
    _hero()

    detection, awareness_tab, performance, about = st.tabs(
        [
            "Message detection",
            "Cyber awareness",
            "Model performance",
            "About & privacy",
        ]
    )
    with detection:
        _detection_tab()
    with awareness_tab:
        _awareness_tab()
    with performance:
        _performance_tab()
    with about:
        _about_tab()

    with st.sidebar:
        st.markdown("### Quick facts")
        st.markdown(
            "- Trained on 5,144 unique SMS after de-duplication\n"
            "- Binary classification: ham / spam\n"
            "- English text only\n"
            "- Threshold chosen on validation data\n"
            "- Scores are uncalibrated"
        )
        st.divider()
        st.caption(
            "If money has already been transferred, call **1930** and file a "
            "complaint at cybercrime.gov.in."
        )


if __name__ == "__main__":
    main()
