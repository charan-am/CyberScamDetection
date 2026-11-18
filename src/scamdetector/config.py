"""Project-wide constants.

This module holds paths, seeds and hyper-parameters only. It contains no logic,
so it can be imported by every other module without creating cycles.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTENT_DIR = PROJECT_ROOT / "content"

RAW_DATASET = RAW_DIR / "SMSSpamCollection"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "scam_clf.joblib"
MODEL_CHECKSUM_PATH = MODELS_DIR / "scam_clf.joblib.sha256"

METRICS_PATH = REPORTS_DIR / "metrics.json"
BASELINE_CSV = REPORTS_DIR / "baseline_comparison.csv"
CONFUSION_CSV = REPORTS_DIR / "confusion_matrix.csv"
CONFLICT_REPORT_PATH = REPORTS_DIR / "conflict_report.json"
SPLIT_SUMMARY_PATH = REPORTS_DIR / "split_summary.json"

AWARENESS_EN_PATH = CONTENT_DIR / "awareness_en.json"
AWARENESS_KN_PATH = CONTENT_DIR / "awareness_kn.json"
REPORTING_PATH = CONTENT_DIR / "reporting.json"

SEED = 42
LABELS = ("ham", "spam")
POSITIVE_LABEL = "spam"

SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# Expected shape of the official UCI release, used only for reporting and
# for the get_data.py sanity check. Never used to re-label or drop rows.
EXPECTED_RAW_ROWS = 5574
EXPECTED_LABEL_COUNTS = {"ham": 4827, "spam": 747}
DATASET_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"

TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 20_000
TFIDF_SUBLINEAR_TF = True
NGRAM_CHOICES = ((1, 1), (1, 2))
C_GRID = (0.1, 1.0, 10.0)
MAX_ITER = 1000
CV_FOLDS = 5

# Threshold is chosen on the validation split only, maximising F-beta with
# beta > 1 so recall is weighted above precision.
THRESHOLD_BETA = 2.0
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEPS = 91

TOP_K_TERMS = 8
MAX_MESSAGE_CHARS = 5000
MIN_MESSAGE_CHARS = 1

ARTIFACT_MAX_BYTES = 10 * 1024 * 1024

# Keyword used by the single-keyword baseline. Chosen because it is the most
# frequent token in the spam class of this corpus; it is a weak rule by design.
BASELINE_KEYWORDS = ("free", "win", "claim", "prize", "urgent", "txt", "call")

URGENCY_TERMS = (
    "urgent",
    "immediately",
    "last chance",
    "expires",
    "expiring",
    "final notice",
    "act now",
    "limited time",
    "verify now",
    "suspend",
    "suspended",
    "blocked",
    "deactivated",
)

PRIZE_TERMS = (
    "winner",
    "won",
    "prize",
    "lottery",
    "lucky draw",
    "jackpot",
    "reward",
    "cash prize",
    "congratulations",
)

OTP_TERMS = ("otp", "one time password", "verification code", "pin", "cvv", "password")

PAYMENT_TERMS = (
    "upi",
    "gpay",
    "google pay",
    "phonepe",
    "paytm",
    "neft",
    "imps",
    "bank account",
    "kyc",
    "refund",
    "cashback",
)

JOB_TERMS = (
    "work from home",
    "part time job",
    "earn daily",
    "earn per day",
    "registration fee",
    "joining fee",
    "no experience",
    "data entry",
)

INVESTMENT_TERMS = (
    "guaranteed return",
    "guaranteed profit",
    "double your money",
    "investment plan",
    "trading tips",
    "crypto",
    "forex",
    "sure shot",
)

MAX_MESSAGE_CHARS_DISPLAY = 5000
