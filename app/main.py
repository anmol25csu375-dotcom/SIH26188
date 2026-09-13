import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR


# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_PATH = REPORTS_DIR / "screening_records.db"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="SIH26188 | Document Screening Portal",
    page_icon="🛡️",
    layout="wide",
)


# ==================================================
# OCR ENGINE
# ==================================================
@st.cache_resource
def load_ocr():
    return PaddleOCR(lang="en")


ocr = load_ocr()


# ==================================================
# FACE MODELS
# ==================================================
@st.cache_resource
def load_face_models():
    if not YUNET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_MODEL_PATH}"
        )

    if not SFACE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SFace model not found: {SFACE_MODEL_PATH}"
        )

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "This OpenCV build does not provide FaceDetectorYN. "
            "Use opencv-contrib-python==4.10.0.84."
        )

    if not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError(
            "This OpenCV build does not provide FaceRecognizerSF. "
            "Use opencv-contrib-python==4.10.0.84."
        )

    face_detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )

    face_recognizer = cv2.FaceRecognizerSF.create(
        str(SFACE_MODEL_PATH),
        "",
    )

    return face_detector, face_recognizer


# ==================================================
# SQLITE STORAGE / REPORTING
# ==================================================
def initialize_database():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS screening_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                document_filename TEXT NOT NULL,
                document_type TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                ocr_confidence REAL NOT NULL,
                face_similarity REAL,
                face_category TEXT,
                classification_risk REAL NOT NULL,
                validation_risk REAL NOT NULL,
                ocr_risk REAL NOT NULL,
                tampering_risk REAL NOT NULL,
                face_risk REAL NOT NULL,
                reasons_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_screening_record(
    document_filename,
    document_type,
    risk_result,
    average_ocr_confidence,
    face_result,
):
    comparison = (face_result or {}).get("comparison") or {}
    breakdown = risk_result["breakdown"]

    face_similarity = comparison.get("cosine_similarity")
    face_category = comparison.get("category")

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO screening_records (
                created_at, document_filename, document_type, risk_score,
                risk_level, ocr_confidence, face_similarity, face_category,
                classification_risk, validation_risk, ocr_risk,
                tampering_risk, face_risk, reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                document_filename,
                document_type,
                float(risk_result["score"]),
                risk_result["level"],
                float(average_ocr_confidence),
                float(face_similarity) if face_similarity is not None else None,
                face_category,
                float(breakdown["Document classification"]),
                float(breakdown["Validation"]),
                float(breakdown["OCR confidence"]),
                float(breakdown["Tampering/image indicators"]),
                float(breakdown["Face similarity"]),
                json.dumps(risk_result["reasons"], ensure_ascii=False),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_recent_screenings(limit=10):
    initialize_database()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, created_at, document_filename, document_type,
                   risk_score, risk_level, ocr_confidence,
                   face_similarity, face_category
            FROM screening_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def build_report_payload(
    document_filename,
    document_type,
    document_type_name,
    average_ocr_confidence,
    classification,
    validation_results,
    tampering_indicators,
    face_result,
    risk_result,
):
    comparison = (face_result or {}).get("comparison") or {}

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "document_filename": document_filename,
        "document_type": document_type,
        "document_type_name": document_type_name,
        "average_ocr_confidence": round(float(average_ocr_confidence), 4),
        "classification": classification,
        "validation": validation_results,
        "tampering_indicators": tampering_indicators,
        "face_similarity": comparison,
        "risk_score": risk_result,
        "final_recommendation": (
            "Human review recommended. This prototype does not automatically "
            "declare a document or identity genuine or fraudulent."
        ),
    }


initialize_database()


# ==================================================
# IMAGE QUALITY
# ==================================================
def calculate_blur_score(image_rgb):
    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )
    laplacian = cv2.Laplacian(
        gray_image,
        cv2.CV_64F,
    )
    return float(laplacian.var())


def calculate_brightness(image_rgb):
    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )
    return float(np.mean(gray_image))


def calculate_contrast(image_rgb):
    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )
    return float(np.std(gray_image))


def calculate_edge_density(image_rgb):
    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )
    edges = cv2.Canny(
        gray_image,
        100,
        200,
    )
    return float(np.count_nonzero(edges) / edges.size)


def calculate_local_inconsistency(image_rgb):
    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    height, width = gray_image.shape
    if height < 4 or width < 4:
        return 0.0

    gray_float = gray_image.astype(np.float32)
    half_h = height // 2
    half_w = width // 2

    regions = [
        gray_float[:half_h, :half_w],
        gray_float[:half_h, half_w:],
        gray_float[half_h:, :half_w],
        gray_float[half_h:, half_w:],
    ]

    means = [float(np.mean(region)) for region in regions if region.size]
    if len(means) < 2:
        return 0.0

    return float(np.std(means))


# ==================================================
# OCR
# ==================================================
def extract_raw_ocr(file_bytes):
    temporary_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_file_path = temporary_file.name

        return ocr.predict(temporary_file_path)
    finally:
        if temporary_file_path and os.path.exists(temporary_file_path):
            os.remove(temporary_file_path)


def polygon_to_rectangle(polygon):
    if polygon is None or len(polygon) == 0:
        return None

    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]

    x_min = min(x_values)
    y_min = min(y_values)
    x_max = max(x_values)
    y_max = max(y_values)

    return {
        "x": int(x_min),
        "y": int(y_min),
        "width": int(x_max - x_min),
        "height": int(y_max - y_min),
    }


def clean_ocr_data(raw_result):
    cleaned_data = []

    for page in raw_result:
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polygons = page.get("rec_polys", [])

        for index, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue

            confidence = None
            if index < len(scores):
                confidence = float(scores[index])

            polygon = None
            if index < len(polygons):
                polygon = polygons[index]

            if polygon is not None and hasattr(polygon, "tolist"):
                polygon = polygon.tolist()

            rectangle = polygon_to_rectangle(polygon)
            if rectangle is None:
                continue

            cleaned_data.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "x": rectangle["x"],
                    "y": rectangle["y"],
                    "width": rectangle["width"],
                    "height": rectangle["height"],
                }
            )

    return cleaned_data


# ==================================================
# TEXT RULES
# ==================================================
PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$", re.IGNORECASE)


def normalize_alphanumeric(text):
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def normalize_digits(text):
    return re.sub(r"\D", "", text)


def looks_like_aadhaar_candidate(text):
    return len(normalize_digits(text)) == 12


def looks_like_pan(text):
    return PAN_PATTERN.fullmatch(normalize_alphanumeric(text)) is not None


def looks_like_license_number(text):
    cleaned = normalize_alphanumeric(text)

    if len(cleaned) < 7 or len(cleaned) > 20:
        return False

    has_letters = any(character.isalpha() for character in cleaned)
    has_digits = any(character.isdigit() for character in cleaned)

    if not (has_letters and has_digits):
        return False

    return cleaned.startswith("DL")


def looks_like_date(text):
    pattern = (
        r"\b"
        r"\d{1,2}"
        r"[\-/\.]"
        r"\d{1,2}"
        r"[\-/\.]"
        r"\d{4}"
        r"\b"
    )
    return re.search(pattern, text) is not None


def extract_date(text):
    pattern = (
        r"\b"
        r"\d{1,2}"
        r"[\-/\.]"
        r"\d{1,2}"
        r"[\-/\.]"
        r"\d{4}"
        r"\b"
    )
    match = re.search(pattern, text)
    return match.group(0) if match else None


def looks_like_gender(text):
    return re.search(
        r"\b(MALE|FEMALE|OTHER)\b",
        text,
        re.IGNORECASE,
    ) is not None


def extract_gender(text):
    match = re.search(
        r"\b(MALE|FEMALE|OTHER)\b",
        text,
        re.IGNORECASE,
    )
    return match.group(1).title() if match else None


def looks_like_id_number(text):
    cleaned = re.sub(r"[\s\-]", "", text)

    if not cleaned.isdigit():
        return False

    return 8 <= len(cleaned) <= 20


def clean_id_number(text):
    return re.sub(r"[\s\-]", "", text)


def find_name_by_label(cleaned_ocr_data):
    """Find a name immediately after a Name label when present."""
    for index, item in enumerate(cleaned_ocr_data):
        label = item["text"].strip().lower().rstrip(":")
        if label not in {"name", "name of holder", "holder name"}:
            continue

        possible_items = cleaned_ocr_data[index + 1:index + 4]
        for candidate in possible_items:
            candidate_text = candidate["text"].strip()
            if looks_like_name(candidate_text):
                result = candidate.copy()
                result["clean_value"] = candidate_text
                return result

    return None


def looks_like_name(text):
    normalized = text.strip()

    if len(normalized) < 2:
        return False

    if looks_like_gender(normalized) or looks_like_date(normalized):
        return False

    letters = sum(character.isalpha() for character in normalized)
    digits = sum(character.isdigit() for character in normalized)

    if letters == 0 or digits > letters:
        return False

    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z]+", normalized)
    }

    # Common non-name words. SAMPLE itself is allowed so synthetic names
    # such as "SAMPLE NAME" can be used in the project demo.
    ignored_words = {
        "government",
        "india",
        "male",
        "female",
        "other",
        "dob",
        "date",
        "birth",
        "address",
        "identity",
        "proof",
        "card",
        "license",
        "father",
        "fathername",
        "fathers",
        "signature",
        "number",
        "nationality",
    }

    if words.intersection(ignored_words):
        return False

    if words in ({"sample", "father"}, {"sample", "father", "name"}):
        return False

    return True


def get_candidate_dates(cleaned_ocr_data):
    candidates = []

    for item in cleaned_ocr_data:
        if looks_like_date(item["text"]):
            candidate = item.copy()
            candidate["clean_value"] = extract_date(item["text"])
            candidates.append(candidate)

    return candidates


def find_name_near_date(cleaned_ocr_data, date_candidate):
    if date_candidate is None:
        return None

    label_name = find_name_by_label(cleaned_ocr_data)
    if label_name is not None:
        return label_name

    date_y = date_candidate["y"]
    date_center_x = date_candidate["x"] + date_candidate["width"] / 2
    possible_names = []

    for item in cleaned_ocr_data:
        if item is date_candidate:
            continue

        if item["y"] >= date_y:
            continue

        if not looks_like_name(item["text"]):
            continue

        item_center_x = item["x"] + item["width"] / 2
        vertical_distance = date_y - item["y"]
        horizontal_distance = abs(item_center_x - date_center_x)

        if vertical_distance > 300:
            continue

        candidate = item.copy()
        candidate["name_layout_score"] = (
            vertical_distance + 0.2 * horizontal_distance
        )
        possible_names.append(candidate)

    if not possible_names:
        return None

    possible_names.sort(key=lambda item: item["name_layout_score"])
    selected = possible_names[0]
    selected["clean_value"] = selected["text"]
    return selected


# ==================================================
# DOCUMENT CLASSIFICATION
# ==================================================
def classify_document(cleaned_ocr_data):
    aadhaar_score = 0
    pan_score = 0
    license_score = 0

    aadhaar_evidence = []
    pan_evidence = []
    license_evidence = []

    date_candidates = get_candidate_dates(cleaned_ocr_data)
    gender_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_gender(item["text"])
    ]

    pan_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_pan(item["text"])
    ]
    aadhaar_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_aadhaar_candidate(item["text"])
    ]
    license_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_license_number(item["text"])
    ]

    if aadhaar_candidates:
        aadhaar_score += 2
        aadhaar_evidence.append("12-digit numeric identifier candidate detected.")

    if date_candidates:
        aadhaar_score += 1
        aadhaar_evidence.append("Date candidate detected.")

    if gender_candidates:
        aadhaar_score += 1
        aadhaar_evidence.append("Gender candidate detected.")

    if date_candidates:
        name_candidate = find_name_near_date(
            cleaned_ocr_data,
            date_candidates[0],
        )
        if name_candidate:
            aadhaar_score += 1
            aadhaar_evidence.append("Name-like text found above or near a date.")

    if pan_candidates:
        pan_score += 3
        pan_evidence.append("10-character PAN-format candidate detected.")

    if date_candidates:
        pan_score += 1
        pan_evidence.append("Date candidate detected.")

        pan_name_candidate = find_name_near_date(
            cleaned_ocr_data,
            date_candidates[0],
        )
        if pan_name_candidate:
            pan_score += 1
            pan_evidence.append("Name-like text found near the date.")

    if license_candidates:
        license_score += 2
        license_evidence.append(
            "Driving-license-like identifier candidate detected."
        )

    if len(date_candidates) >= 2:
        license_score += 2
        license_evidence.append("Multiple date candidates detected.")

    if date_candidates:
        license_name_candidate = find_name_near_date(
            cleaned_ocr_data,
            date_candidates[0],
        )
        if license_name_candidate:
            license_score += 1
            license_evidence.append("Name-like text found above a date.")

    scores = {
        "aadhaar_like": aadhaar_score,
        "pan_like": pan_score,
        "license_like": license_score,
    }

    evidence = {
        "aadhaar_like": aadhaar_evidence,
        "pan_like": pan_evidence,
        "license_like": license_evidence,
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score < 3:
        document_type = "unknown"
    elif best_score == second_score:
        document_type = "review"
    elif best_score - second_score < 1:
        document_type = "review"
    else:
        document_type = best_type

    return {
        "document_type": document_type,
        "score": best_score,
        "scores": scores,
        "evidence": evidence,
    }


def get_document_type_name(document_type):
    names = {
        "aadhaar_like": "Aadhaar-like Document",
        "pan_like": "PAN-like Document",
        "license_like": "Driving License-like Document",
        "review": "Needs Review",
        "unknown": "Unknown Document",
    }
    return names.get(document_type, "Unknown Document")


# ==================================================
# FIELD EXTRACTION
# ==================================================
def find_common_candidates(cleaned_ocr_data):
    date_candidates = []
    gender_candidates = []
    id_candidates = []

    for item in cleaned_ocr_data:
        text = item["text"]

        if looks_like_date(text):
            candidate = item.copy()
            candidate["clean_value"] = extract_date(text)
            date_candidates.append(candidate)

        if looks_like_gender(text):
            candidate = item.copy()
            candidate["clean_value"] = extract_gender(text)
            gender_candidates.append(candidate)

        if looks_like_id_number(text):
            candidate = item.copy()
            candidate["clean_value"] = clean_id_number(text)
            id_candidates.append(candidate)

    return date_candidates, gender_candidates, id_candidates


def extract_aadhaar_fields(cleaned_ocr_data):
    date_candidates, gender_candidates, _ = find_common_candidates(
        cleaned_ocr_data
    )

    aadhaar_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_aadhaar_candidate(item["text"])
    ]

    name_candidate = (
        find_name_near_date(cleaned_ocr_data, date_candidates[0])
        if date_candidates
        else find_name_by_label(cleaned_ocr_data)
    )

    id_candidate = None
    if aadhaar_candidates:
        id_candidate = aadhaar_candidates[0].copy()
        id_candidate["clean_value"] = normalize_digits(
            id_candidate["text"]
        )

    return {
        "name": name_candidate,
        "dob": date_candidates[0] if date_candidates else None,
        "gender": gender_candidates[0] if gender_candidates else None,
        "id_number": id_candidate,
    }


def extract_pan_fields(cleaned_ocr_data):
    date_candidates = get_candidate_dates(cleaned_ocr_data)
    pan_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_pan(item["text"])
    ]

    name_candidate = (
        find_name_near_date(cleaned_ocr_data, date_candidates[0])
        if date_candidates
        else find_name_by_label(cleaned_ocr_data)
    )

    pan_candidate = None
    if pan_candidates:
        pan_candidate = pan_candidates[0].copy()
        pan_candidate["clean_value"] = normalize_alphanumeric(
            pan_candidate["text"]
        )

    return {
        "name": name_candidate,
        "dob": date_candidates[0] if date_candidates else None,
        "gender": None,
        "id_number": pan_candidate,
    }


def extract_license_fields(cleaned_ocr_data):
    date_candidates = get_candidate_dates(cleaned_ocr_data)
    license_candidates = [
        item for item in cleaned_ocr_data
        if looks_like_license_number(item["text"])
    ]

    name_candidate = (
        find_name_near_date(cleaned_ocr_data, date_candidates[0])
        if date_candidates
        else find_name_by_label(cleaned_ocr_data)
    )

    license_candidate = None
    if license_candidates:
        license_candidate = license_candidates[0].copy()
        license_candidate["clean_value"] = normalize_alphanumeric(
            license_candidate["text"]
        )

    return {
        "name": name_candidate,
        "dob": date_candidates[0] if date_candidates else None,
        "gender": None,
        "id_number": license_candidate,
    }


def extract_fields_by_document_type(document_type, cleaned_ocr_data):
    if document_type == "aadhaar_like":
        return extract_aadhaar_fields(cleaned_ocr_data)

    if document_type == "pan_like":
        return extract_pan_fields(cleaned_ocr_data)

    if document_type == "license_like":
        return extract_license_fields(cleaned_ocr_data)

    return {
        "name": None,
        "dob": None,
        "gender": None,
        "id_number": None,
    }


def create_document_profile(document_type, fields):
    profile = {
        "document_type": document_type,
        "name": None,
        "dob": None,
        "gender": None,
        "id_number": None,
        "confidence": {
            "name": None,
            "dob": None,
            "gender": None,
            "id_number": None,
        },
    }

    for field_name in ["name", "dob", "gender", "id_number"]:
        field = fields.get(field_name)
        if field:
            profile[field_name] = field["clean_value"]
            profile["confidence"][field_name] = field["confidence"]

    return profile


# ==================================================
# VALIDATION
# ==================================================
def parse_date(value):
    if not value:
        return None

    for date_format in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            pass

    return None


def validate_fields(document_profile, document_type):
    results = []

    required_fields = {
        "aadhaar_like": ["name", "dob", "gender", "id_number"],
        "pan_like": ["name", "dob", "id_number"],
        "license_like": ["name", "dob", "id_number"],
    }.get(document_type, [])

    for field_name in required_fields:
        found = bool(document_profile.get(field_name))
        results.append(
            {
                "name": f"Required field: {field_name}",
                "passed": found,
                "message": (
                    "Field detected."
                    if found
                    else "Field was not detected. Manual review is recommended."
                ),
            }
        )

    dob = parse_date(document_profile.get("dob"))
    dob_valid = dob is not None and dob <= datetime.now().date()
    if document_profile.get("dob"):
        results.append(
            {
                "name": "DOB format/date",
                "passed": dob_valid,
                "message": (
                    "DOB parsed and is not in the future."
                    if dob_valid
                    else "DOB could not be validated as a sensible past date."
                ),
            }
        )

    id_number = document_profile.get("id_number") or ""
    id_valid = True
    id_message = "No type-specific identifier validation available."

    if document_type == "pan_like":
        id_valid = PAN_PATTERN.fullmatch(id_number) is not None
        id_message = (
            "PAN-like format matches the prototype pattern."
            if id_valid
            else "PAN-like identifier format does not match the prototype pattern."
        )
    elif document_type == "aadhaar_like":
        id_valid = len(normalize_digits(id_number)) == 12
        id_message = (
            "12-digit identifier candidate matches the prototype rule."
            if id_valid
            else "Identifier does not match the 12-digit prototype rule."
        )
    elif document_type == "license_like":
        id_valid = looks_like_license_number(id_number)
        id_message = (
            "Synthetic license-like identifier matches the prototype rule."
            if id_valid
            else "License-like identifier does not match the prototype rule."
        )

    if document_type != "unknown":
        results.append(
            {
                "name": "Identifier format",
                "passed": id_valid,
                "message": id_message,
            }
        )

    return results


# ==================================================
# TAMPERING INDICATORS
# ==================================================
def run_tampering_indicators(image_rgb):
    blur_score = calculate_blur_score(image_rgb)
    brightness = calculate_brightness(image_rgb)
    contrast = calculate_contrast(image_rgb)
    edge_density = calculate_edge_density(image_rgb)
    local_inconsistency = calculate_local_inconsistency(image_rgb)

    indicators = []

    indicators.append(
        {
            "name": "Blur",
            "value": blur_score,
            "status": "REVIEW" if blur_score < 100 else "PASS",
            "message": (
                "Image may be blurry."
                if blur_score < 100
                else "Image has reasonable sharpness."
            ),
        }
    )

    indicators.append(
        {
            "name": "Brightness",
            "value": brightness,
            "status": (
                "REVIEW"
                if brightness < 40 or brightness > 220
                else "PASS"
            ),
            "message": (
                "Image may be unusually dark or bright."
                if brightness < 40 or brightness > 220
                else "Brightness is within a broad screening range."
            ),
        }
    )

    indicators.append(
        {
            "name": "Contrast",
            "value": contrast,
            "status": "REVIEW" if contrast < 20 else "PASS",
            "message": (
                "Image contrast is low."
                if contrast < 20
                else "Image has usable contrast."
            ),
        }
    )

    indicators.append(
        {
            "name": "Edge density",
            "value": edge_density,
            "status": "INFO",
            "message": "Edge density is recorded as a screening signal.",
        }
    )

    indicators.append(
        {
            "name": "Local inconsistency",
            "value": local_inconsistency,
            "status": "INFO",
            "message": "Quadrant brightness variation is recorded as a screening signal.",
        }
    )

    return indicators


# ==================================================
# OCR VISUALIZATION
# ==================================================
def draw_ocr_boxes(image_rgb, cleaned_ocr_data):
    output_image = image_rgb.copy()

    for item in cleaned_ocr_data:
        x = item["x"]
        y = item["y"]
        width = item["width"]
        height = item["height"]

        cv2.rectangle(
            output_image,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2,
        )

    return output_image


# ==================================================
# FACE DETECTION
# ==================================================
def image_bytes_to_rgb(file_bytes):
    pil_image = Image.open(tempfile.SpooledTemporaryFile()).convert("RGB")
    return np.array(pil_image)


def decode_image_bytes(file_bytes):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError("Could not decode the uploaded image.")

    return bgr_image


def detect_faces(bgr_image, face_detector):
    height, width = bgr_image.shape[:2]
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(bgr_image)

    if faces is None:
        return []

    return [face.astype(np.float32) for face in faces]


def draw_yunet_faces(bgr_image, faces):
    output = cv2.cvtColor(bgr_image.copy(), cv2.COLOR_BGR2RGB)

    for index, face in enumerate(faces, start=1):
        x, y, width, height = face[:4]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(output.shape[1] - 1, int(x + width))
        y2 = min(output.shape[0] - 1, int(y + height))

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"Face {index}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output


def face_quality_check(bgr_image, face):
    """
    Check whether a detected face is large enough for SFace screening.

    Laplacian variance is kept as an advisory sharpness signal instead of
    a hard blocker. Face crops from photographs, compressed images, or
    printed/embedded document portraits can legitimately have low
    Laplacian scores while still being usable by the face recognizer.
    """
    x, y, width, height = [int(round(value)) for value in face[:4]]

    x = max(0, x)
    y = max(0, y)
    width = max(0, width)
    height = max(0, height)

    x2 = min(bgr_image.shape[1], x + width)
    y2 = min(bgr_image.shape[0], y + height)

    if x2 <= x or y2 <= y:
        return {
            "passed": False,
            "width": 0,
            "height": 0,
            "blur_score": 0.0,
            "sharpness_warning": True,
            "message": "Detected face crop is invalid.",
        }

    crop = bgr_image[y:y2, x:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    blur_score = calculate_blur_score(crop_rgb)

    passed_size = crop.shape[1] >= 80 and crop.shape[0] >= 80
    sharpness_warning = blur_score < 10
    passed = passed_size

    if not passed_size:
        message = "Face region is too small for reliable screening."
    elif sharpness_warning:
        message = (
            "Face crop is somewhat soft, but it is large enough for SFace "
            "screening. Sharpness is treated as a warning, not a hard blocker."
        )
    else:
        message = "Face crop is large enough for SFace screening."

    return {
        "passed": passed,
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
        "blur_score": float(blur_score),
        "sharpness_warning": sharpness_warning,
        "message": message,
    }


def extract_face_features(bgr_image, face, face_recognizer):
    aligned_face = face_recognizer.alignCrop(
        bgr_image,
        face,
    )
    features = face_recognizer.feature(aligned_face)
    return features


def compare_face_features(
    document_features,
    separate_features,
    face_recognizer,
):
    score = float(
        face_recognizer.match(
            document_features,
            separate_features,
            cv2.FaceRecognizerSF_FR_COSINE,
        )
    )

    if score >= 0.45:
        category = "HIGH_SIMILARITY"
    elif score >= 0.363:
        category = "REFERENCE_MATCH"
    elif score >= 0.30:
        category = "UNCERTAIN"
    else:
        category = "LOW_SIMILARITY"

    return {
        "cosine_similarity": score,
        "category": category,
        "reference_threshold": 0.363,
    }


def rank_document_face_candidates(bgr_image, faces):
    """
    Rank document face candidates so a faint or decorative face is less
    likely to be selected automatically. The ranking uses detector
    confidence and face size first, with sharpness as a supporting signal.
    """

    ranked = []

    image_area = float(bgr_image.shape[0] * bgr_image.shape[1])
    if image_area <= 0:
        return []

    for index, face in enumerate(faces):
        x, y, width, height = [int(round(value)) for value in face[:4]]

        area = max(0, width) * max(0, height)
        area_ratio = area / image_area
        confidence = float(face[14]) if len(face) > 14 else 0.0

        quality = face_quality_check(
            bgr_image,
            face,
        )

        # A larger, higher-confidence face is normally the actual
        # document portrait. Sharpness helps down-rank faint ghost images.
        size_score = min(1.0, area_ratio / 0.08)
        blur_score = float(quality["blur_score"])
        sharpness_score = min(1.0, blur_score / 150.0)

        selection_score = (
            0.50 * confidence
            + 0.35 * size_score
            + 0.15 * sharpness_score
        )

        ranked.append(
            {
                "index": index,
                "face": face,
                "confidence": confidence,
                "area": area,
                "area_ratio": area_ratio,
                "quality": quality,
                "selection_score": selection_score,
            }
        )

    ranked.sort(
        key=lambda item: item["selection_score"],
        reverse=True,
    )

    return ranked


def run_face_verification(
    document_bgr,
    separate_bgr,
    face_detector,
    face_recognizer,
    selected_document_index=None,
):
    document_faces = detect_faces(document_bgr, face_detector)
    separate_faces = detect_faces(separate_bgr, face_detector)

    result = {
        "document_faces": document_faces,
        "separate_faces": separate_faces,
        "document_ranked_faces": [],
        "selected_document_index": selected_document_index,
        "document_quality": None,
        "separate_quality": None,
        "comparison": None,
        "status": "REVIEW",
        "message": "Manual review is recommended.",
    }

    if not document_faces:
        result["message"] = (
            "No document face was detected. Face-similarity screening cannot be completed."
        )
        return result

    result["document_ranked_faces"] = rank_document_face_candidates(
        document_bgr,
        document_faces,
    )

    if selected_document_index is None:
        selected_document_index = result["document_ranked_faces"][0]["index"]

    if selected_document_index < 0 or selected_document_index >= len(document_faces):
        result["message"] = "Invalid document-face selection."
        return result

    result["selected_document_index"] = selected_document_index

    if len(separate_faces) != 1:
        result["message"] = (
            "The separate face image must contain exactly one detectable face."
        )
        return result

    document_face = document_faces[selected_document_index]
    separate_face = separate_faces[0]

    result["document_quality"] = face_quality_check(
        document_bgr,
        document_face,
    )
    result["separate_quality"] = face_quality_check(
        separate_bgr,
        separate_face,
    )

    if not result["document_quality"]["passed"]:
        result["message"] = (
            "The selected document face does not pass the basic size and sharpness checks. "
            "Try another detected face or use a clearer document image."
        )
        return result

    if not result["separate_quality"]["passed"]:
        result["message"] = (
            "Separate face quality is insufficient for reliable screening."
        )
        return result

    document_features = extract_face_features(
        document_bgr,
        document_face,
        face_recognizer,
    )
    separate_features = extract_face_features(
        separate_bgr,
        separate_face,
        face_recognizer,
    )

    result["comparison"] = compare_face_features(
        document_features,
        separate_features,
        face_recognizer,
    )

    result["status"] = "PASS"
    result["message"] = (
        "Similarity score calculated. This is a screening signal, not an identity verdict."
    )

    return result


# ==================================================
# DISPLAY HELPERS
# ==================================================
def show_status(status, message):
    if status == "PASS":
        st.success(message)
    elif status == "INFO":
        st.info(message)
    else:
        st.warning(message)


# ==================================================
# EXPLAINABLE RISK SCORE
# ==================================================
def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def calculate_average_ocr_confidence(cleaned_ocr_data):
    confidence_values = [
        float(item["confidence"])
        for item in cleaned_ocr_data
        if item.get("confidence") is not None
    ]

    if not confidence_values:
        return 0.0

    return sum(confidence_values) / len(confidence_values)


def calculate_risk_score(
    document_type,
    classification,
    validation_results,
    tampering_indicators,
    average_ocr_confidence,
    face_result,
):
    """
    Calculate an explainable 0-100 screening risk score.

    The score measures review concerns, not fraud probability.
    Passing components contribute 0 risk; failed/uncertain components add risk.
    """
    breakdown = {}
    reasons = []

    # 1. Document classification: 10 points
    if document_type == "unknown":
        classification_risk = 10.0
        reasons.append("Document type could not be classified reliably.")
    elif document_type == "review":
        classification_risk = 7.0
        reasons.append("Document type classification is ambiguous.")
    else:
        classification_risk = 0.0

    breakdown["Document classification"] = classification_risk

    # 2. Validation: 30 points
    if validation_results:
        failed_validations = sum(
            1 for check in validation_results if not check["passed"]
        )
        validation_risk = (
            failed_validations / len(validation_results)
        ) * 30.0

        for check in validation_results:
            if not check["passed"]:
                reasons.append(
                    f"Validation review needed: {check['name']}."
                )
    else:
        validation_risk = 15.0
        reasons.append(
            "No validation result was available for the document."
        )

    breakdown["Validation"] = validation_risk

    # 3. OCR confidence: 20 points
    average_ocr_confidence = clamp(average_ocr_confidence)
    ocr_risk = (1.0 - average_ocr_confidence) * 20.0
    if average_ocr_confidence < 0.70:
        reasons.append(
            f"Average OCR confidence is relatively low ({average_ocr_confidence:.3f})."
        )
    breakdown["OCR confidence"] = ocr_risk

    # 4. Image/tampering indicators: 20 points
    reviewable_indicators = [
        indicator
        for indicator in tampering_indicators
        if indicator["status"] in {"PASS", "REVIEW"}
    ]
    review_count = sum(
        1 for indicator in reviewable_indicators
        if indicator["status"] == "REVIEW"
    )

    if reviewable_indicators:
        tampering_risk = (
            review_count / len(reviewable_indicators)
        ) * 20.0
    else:
        tampering_risk = 10.0
        reasons.append(
            "No image-quality/tampering indicator result was available."
        )

    for indicator in reviewable_indicators:
        if indicator["status"] == "REVIEW":
            reasons.append(
                f"Image-quality review needed: {indicator['name']}."
            )

    breakdown["Tampering/image indicators"] = tampering_risk

    # 5. Face similarity: 20 points
    face_category = None
    face_comparison = (face_result or {}).get("comparison") if face_result else None

    if face_comparison:
        face_category = face_comparison.get("category")
        category_risk = {
            "HIGH_SIMILARITY": 0.0,
            "REFERENCE_MATCH": 4.0,
            "UNCERTAIN": 10.0,
            "LOW_SIMILARITY": 20.0,
        }
        face_risk = category_risk.get(face_category, 10.0)

        if face_category == "LOW_SIMILARITY":
            reasons.append(
                "Face similarity produced a lower similarity signal."
            )
        elif face_category == "UNCERTAIN":
            reasons.append(
                "Face similarity is in an uncertain range."
            )
        elif face_category == "REFERENCE_MATCH":
            reasons.append(
                "Face similarity reached the reference-match range; this remains only a screening signal."
            )
    else:
        face_risk = 10.0
        if face_result is None:
            reasons.append(
                "Face similarity was not run because the separate face photo was unavailable."
            )
        else:
            reasons.append(
                "Face similarity could not produce a comparison."
            )

    breakdown["Face similarity"] = face_risk

    total_risk = round(
        min(100.0, sum(breakdown.values())),
        2,
    )

    if total_risk < 30:
        level = "LOW"
    elif total_risk < 60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "score": total_risk,
        "level": level,
        "breakdown": breakdown,
        "reasons": reasons,
        "weights": {
            "Document classification": 10,
            "Validation": 30,
            "OCR confidence": 20,
            "Tampering/image indicators": 20,
            "Face similarity": 20,
        },
    }


# ==================================================
# SITE SHELL
# ==================================================

SITE_CSS = """
<style>
    .stApp {
        background: #0b1020;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #0a0f1c;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .hero {
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(30,41,80,0.96), rgba(15,23,42,0.98));
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 18px 50px rgba(0,0,0,0.22);
        margin-bottom: 20px;
    }
    .hero-kicker {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.72;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.35rem;
        line-height: 1.08;
        font-weight: 800;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        opacity: 0.82;
        margin-top: 12px;
        max-width: 900px;
    }
    .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 16px;
    }
    .pill {
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        font-size: 0.78rem;
    }
    .section-card {
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid rgba(148,163,184,0.15);
        background: rgba(15,23,42,0.72);
        margin: 10px 0;
    }
    .small-muted {
        color: rgba(226,232,240,0.65);
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.15);
        padding: 14px;
        border-radius: 14px;
    }
</style>
"""

st.markdown(SITE_CSS, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🛡️ SIH26188")
    st.caption("AI-Based Fake Identity & Document Screening")
    st.markdown("---")
    site_page = st.radio(
        "Navigate",
        ["Overview", "Screening", "Reports & History", "About"],
        index=0,
    )
    st.markdown("---")
    st.caption("Synthetic/redacted document prototype")
    st.caption("Human review remains the final decision step.")

# Header shared by every page
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Smart India Hackathon 2026 · SIH26188</div>
        <div class="hero-title">AI-Based Fake Identity &amp; Document Screening</div>
        <div class="hero-subtitle">
            A modular screening portal combining OCR, document rules, image-quality indicators,
            face similarity, explainable risk scoring, and review-ready reporting.
        </div>
        <div class="pill-row">
            <span class="pill">📄 OCR</span>
            <span class="pill">🧠 Document Rules</span>
            <span class="pill">🧍 YuNet + SFace</span>
            <span class="pill">📊 Risk Score</span>
            <span class="pill">🗂️ SQLite Reports</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# OVERVIEW PAGE
# ==================================================
if site_page == "Overview":
    st.subheader("A screening workflow built for an SIH demo")
    st.write(
        "Upload a synthetic or redacted document, optionally provide a separate face photo, "
        "and review the system's explainable screening signals in one place."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📄 Document screening")
        st.write("PaddleOCR, document-type clues, structured field extraction, and validation rules.")
    with c2:
        st.markdown("### 🧍 Face similarity")
        st.write("YuNet detects candidate portraits and SFace produces a similarity signal for review.")
    with c3:
        st.markdown("### 📊 Explainable decision support")
        st.write("A transparent 0–100 screening score plus reasons, reports, and saved history.")

    st.markdown("### How it works")
    st.markdown(
        """
        <div class="section-card">
            <b>1. Upload</b> → document + optional face photo<br>
            <b>2. Analyze</b> → OCR + classification + validation + image indicators<br>
            <b>3. Compare</b> → selected document portrait + separate face photo<br>
            <b>4. Score</b> → explainable risk score from 0 to 100<br>
            <b>5. Review</b> → human decision remains final
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "For demonstrations, keep government-ID content synthetic or redacted. "
        "Face testing can use your own photo or a consenting volunteer's photo."
    )

# ==================================================
# REPORTS PAGE
# ==================================================
elif site_page == "Reports & History":
    initialize_database()
    st.subheader("Reports & History")
    records = load_recent_screenings(limit=100)

    if not records:
        st.info("No saved screenings yet. Run a screening and save the result to populate this page.")
    else:
        scores = [float(r["risk_score"]) for r in records]
        low_count = sum(r["risk_level"] == "LOW" for r in records)
        medium_count = sum(r["risk_level"] == "MEDIUM" for r in records)
        high_count = sum(r["risk_level"] == "HIGH" for r in records)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Screenings", len(records))
        m2.metric("Low risk", low_count)
        m3.metric("Medium risk", medium_count)
        m4.metric("High risk", high_count)

        st.markdown("### Recent screenings")
        st.dataframe(records, use_container_width=True, hide_index=True)

        with st.expander("Project storage"):
            st.write(f"SQLite database: `{DATABASE_PATH.relative_to(BASE_DIR)}`")
            st.write(f"Reports folder: `{REPORTS_DIR.relative_to(BASE_DIR)}`")

# ==================================================
# ABOUT PAGE
# ==================================================
elif site_page == "About":
    st.subheader("About this prototype")
    st.markdown(
        """
        **SIH26188** is a beginner-friendly prototype for AI-assisted identity and
        document screening. It is designed as decision support rather than an automatic
        authenticity or identity-verification authority.
        """
    )

    with st.expander("Technology stack", expanded=True):
        st.write("Python · Streamlit · PaddleOCR · OpenCV · YuNet · SFace · NumPy · SQLite")

    with st.expander("Face verification", expanded=True):
        st.write(
            "YuNet is used to detect face candidates. SFace generates face features and "
            "a cosine similarity score. The score is treated as a screening signal only."
        )

    with st.expander("Safety and scope", expanded=True):
        st.write(
            "Use synthetic or redacted identity documents for the project demo. Do not use "
            "the result as a legal determination of authenticity or identity. Human review "
            "must remain the final step."
        )

# ==================================================
# SCREENING PAGE
# ==================================================
else:
    st.subheader("Document screening")
    st.caption(
        "Upload a synthetic or redacted document. Add an optional separate face photo "
        "for YuNet + SFace similarity screening."
    )

    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        uploaded_file = st.file_uploader(
            "📄 Document image",
            type=["jpg", "jpeg", "png"],
            key="document_uploader",
        )
    with upload_col2:
        separate_face_file = st.file_uploader(
            "🧍 Separate face photo (optional)",
            type=["jpg", "jpeg", "png"],
            key="separate_face_uploader",
        )

    st.caption(
        "For the document side of the project, use synthetic/redacted documents. "
        "For face testing, use your own photo or a consenting volunteer's photo."
    )

    if uploaded_file is None:
        st.info("Please upload a synthetic or redacted document image.")
    else:
        # Values used later by the explainable risk score.
        validation_results = []
        tampering_indicators = []
        average_ocr_confidence = 0.0
        classification = {
            "document_type": "unknown",
            "score": 0,
            "scores": {},
            "evidence": {},
        }
        document_type = "unknown"
        document_type_name = get_document_type_name(document_type)
        face_result = None

        # ----------------------------------------------
        # Load document image
        # ----------------------------------------------
        document_image = Image.open(uploaded_file).convert("RGB")
        document_rgb = np.array(document_image)

        st.subheader("2. Document Image")
        st.image(document_image, caption=uploaded_file.name)

        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(f"**Image size:** {document_image.width} × {document_image.height} pixels")

        # ----------------------------------------------
        # Image quality
        # ----------------------------------------------
        st.subheader("3. Image Quality")

        blur_score = calculate_blur_score(document_rgb)
        st.write(f"**Blur score:** {blur_score:.2f}")

        if blur_score < 100:
            st.warning("The document image may be blurry.")
        else:
            st.success("The document image appears reasonably clear.")

        # ----------------------------------------------
        # OCR
        # ----------------------------------------------
        st.subheader("4. OCR")

        try:
            with st.spinner("Reading document with PaddleOCR..."):
                raw_ocr_result = extract_raw_ocr(uploaded_file.getvalue())

            cleaned_ocr_data = clean_ocr_data(raw_ocr_result)
            average_ocr_confidence = calculate_average_ocr_confidence(
                cleaned_ocr_data
            )
        except Exception as exc:
            st.error("OCR failed.")
            st.exception(exc)
            cleaned_ocr_data = []
            average_ocr_confidence = 0.0

        if not cleaned_ocr_data:
            st.warning("No readable text was detected.")
        else:
            st.write(f"**OCR text regions detected:** {len(cleaned_ocr_data)}")

            with st.expander("Show OCR text"):
                for item in cleaned_ocr_data:
                    confidence_text = (
                        f"{item['confidence']:.3f}"
                        if item["confidence"] is not None
                        else "N/A"
                    )
                    st.write(
                        f"{item['text']} — confidence {confidence_text}"
                    )

            # ------------------------------------------
            # Classification
            # ------------------------------------------
            st.subheader("5. Document Classification")

            classification = classify_document(cleaned_ocr_data)
            document_type = classification["document_type"]
            document_type_name = get_document_type_name(document_type)

            if document_type == "unknown":
                st.warning("Unknown document type.")
            elif document_type == "review":
                st.warning(
                    "Document type is ambiguous. Manual review is recommended."
                )
            else:
                st.success(f"Detected: {document_type_name}")

            st.write(f"**Classification score:** {classification['score']}")
            st.write(f"**Type scores:** {classification['scores']}")

            evidence = classification["evidence"].get(document_type, [])
            if evidence:
                st.write("**Classification evidence:**")
                for evidence_item in evidence:
                    st.write(f"- {evidence_item}")

            # ------------------------------------------
            # OCR boxes
            # ------------------------------------------
            boxed_image = draw_ocr_boxes(
                document_rgb,
                cleaned_ocr_data,
            )

            st.subheader("OCR Bounding Boxes")
            st.image(boxed_image, caption="Detected text regions")

            # ------------------------------------------
            # Field extraction
            # ------------------------------------------
            st.subheader("6. Extracted Document Fields")

            fields = extract_fields_by_document_type(
                document_type,
                cleaned_ocr_data,
            )
            document_profile = create_document_profile(
                document_type,
                fields,
            )

            st.caption(
                "Field extraction uses prototype rules and does not prove that the document is genuine."
            )

            st.write(f"**Document Type:** {document_type_name}")
            st.write(f"**Name:** {document_profile['name'] or 'Not found'}")
            st.write(f"**DOB:** {document_profile['dob'] or 'Not found'}")
            st.write(f"**Gender:** {document_profile['gender'] or 'Not found'}")
            st.write(f"**ID Number:** {document_profile['id_number'] or 'Not found'}")

            with st.expander("Field detection details"):
                for field_name in ["name", "dob", "gender", "id_number"]:
                    field = fields[field_name]
                    st.markdown(f"**{field_name.replace('_', ' ').title()}**")

                    if field:
                        st.write(f"Value: {field['clean_value']}")
                        if field["confidence"] is not None:
                            st.write(f"OCR confidence: {field['confidence']:.3f}")
                        st.write(
                            "Position: "
                            f"x={field['x']}, y={field['y']}, "
                            f"width={field['width']}, height={field['height']}"
                        )
                    else:
                        st.write("Not found.")

            # ------------------------------------------
            # Validation
            # ------------------------------------------
            st.subheader("7. Validation Checks")

            validation_results = validate_fields(
                document_profile,
                document_type,
            )

            for check in validation_results:
                show_status(
                    "PASS" if check["passed"] else "REVIEW",
                    f"{'PASS' if check['passed'] else 'REVIEW'} — {check['name']}: {check['message']}",
                )

            # ------------------------------------------
            # Tampering indicators
            # ------------------------------------------
            st.subheader("8. Tampering Indicators")
            st.caption(
                "These are image-quality and consistency indicators only. They do not by themselves detect or prove forgery."
            )

            tampering_indicators = run_tampering_indicators(document_rgb)
            for indicator in tampering_indicators:
                value = indicator["value"]
                if indicator["name"] == "Edge density":
                    value_text = f"{value:.4f}"
                else:
                    value_text = f"{value:.2f}"

                show_status(
                    "PASS" if indicator["status"] == "PASS" else "INFO" if indicator["status"] == "INFO" else "REVIEW",
                    f"{indicator['status']} — {indicator['name']}: {value_text}. {indicator['message']}",
                )

        # ----------------------------------------------
        # Face verification
        # ----------------------------------------------
        st.subheader("9. Face Verification / Similarity Screening")

        if separate_face_file is None:
            st.info(
                "Upload a separate face photo above to run YuNet + SFace similarity screening."
            )
        else:
            try:
                with st.spinner("Loading YuNet and SFace models..."):
                    face_detector, face_recognizer = load_face_models()

                document_bgr = decode_image_bytes(uploaded_file.getvalue())
                separate_bgr = decode_image_bytes(separate_face_file.getvalue())

                # Detect first so the user can distinguish the real document
                # portrait from a faded/ghost/decorative face on the document.
                document_faces = detect_faces(document_bgr, face_detector)
                separate_faces = detect_faces(separate_bgr, face_detector)

                document_ranked = rank_document_face_candidates(
                    document_bgr,
                    document_faces,
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Document face detection**")
                    st.write(f"Faces detected: {len(document_faces)}")

                    document_face_preview = draw_yunet_faces(
                        document_bgr,
                        document_faces,
                    )
                    st.image(
                        document_face_preview,
                        caption="YuNet detection on document",
                    )

                    if len(document_faces) > 1:
                        st.info(
                            "Multiple faces were detected. A faded, ghost, watermark, "
                            "or decorative portrait can also be detected by YuNet. "
                            "Select the actual document portrait below."
                        )

                    if document_ranked:
                        labels = []
                        label_to_index = {}

                        for item in document_ranked:
                            face_index = item["index"]
                            label = (
                                f"Face {face_index + 1} — "
                                f"confidence {item['confidence']:.3f}, "
                                f"selection score {item['selection_score']:.3f}"
                            )
                            labels.append(label)
                            label_to_index[label] = face_index

                        selected_label = st.selectbox(
                            "Document portrait to compare",
                            labels,
                            index=0,
                            help=(
                                "The first option is the automatic ranking result. "
                                "Change it when a faint or ghost portrait was ranked above "
                                "the actual document portrait."
                            ),
                        )

                        selected_document_index = label_to_index[selected_label]
                        selected_ranked_item = next(
                            item
                            for item in document_ranked
                            if item["index"] == selected_document_index
                        )

                        selected_quality = selected_ranked_item["quality"]
                        st.write(
                            f"Selected Face {selected_document_index + 1}: "
                            f"{selected_quality['width']} × {selected_quality['height']} px, "
                            f"blur score {selected_quality['blur_score']:.2f}"
                        )
                    else:
                        selected_document_index = None

                with col2:
                    st.markdown("**Separate face detection**")
                    st.write(f"Faces detected: {len(separate_faces)}")
                    separate_face_preview = draw_yunet_faces(
                        separate_bgr,
                        separate_faces,
                    )
                    st.image(
                        separate_face_preview,
                        caption="YuNet detection on separate photo",
                    )

                    if len(separate_faces) > 1:
                        st.warning(
                            "More than one face was detected in the separate photo. "
                            "Use a photo containing only the intended test face."
                        )

                face_result = run_face_verification(
                    document_bgr,
                    separate_bgr,
                    face_detector,
                    face_recognizer,
                    selected_document_index=selected_document_index,
                )

                if face_result["document_quality"]:
                    st.write("**Selected document face quality**")
                    st.write(face_result["document_quality"])
                    if face_result["document_quality"].get("sharpness_warning"):
                        st.info(
                            "Document face sharpness is a warning only. The crop is still allowed "
                            "to continue to SFace comparison because it is large enough."
                        )

                if face_result["separate_quality"]:
                    st.write("**Separate face quality**")
                    st.write(face_result["separate_quality"])
                    if face_result["separate_quality"].get("sharpness_warning"):
                        st.info(
                            "Separate face sharpness is a warning only. The photo is still allowed "
                            "to continue to SFace comparison because the face is large enough."
                        )

                comparison = face_result["comparison"]

                if comparison:
                    score = comparison["cosine_similarity"]
                    st.write(f"**Cosine similarity:** {score:.4f}")
                    st.write(
                        f"**Reference threshold:** {comparison['reference_threshold']:.3f}"
                    )
                    st.write(f"**Category:** {comparison['category']}")

                    if comparison["category"] == "HIGH_SIMILARITY":
                        st.success(
                            "HIGH SIMILARITY — strong similarity signal for this prototype. Human review is still required."
                        )
                    elif comparison["category"] == "REFERENCE_MATCH":
                        st.success(
                            "REFERENCE MATCH — similarity is at or above the OpenCV reference threshold. Human review is still required."
                        )
                    elif comparison["category"] == "UNCERTAIN":
                        st.warning(
                            "UNCERTAIN — similarity is in an intermediate range. Manual review is recommended."
                        )
                    else:
                        st.warning(
                            "LOW SIMILARITY — the images produce a lower similarity signal. Manual review is recommended."
                        )

                else:
                    st.warning(face_result["message"])

                st.caption(
                    "YuNet is used for face detection and SFace for feature-based similarity. "
                    "The cosine score is a screening signal, not an identity probability or legal verification."
                )

            except Exception as exc:
                st.error("Face verification could not be completed.")
                st.exception(exc)

        # ----------------------------------------------
        # Explainable risk score
        # ----------------------------------------------
        st.subheader("10. Explainable Risk Score")
        st.caption(
            "This score is a transparent screening score from 0 to 100. "
            "Higher scores mean more review signals. It is not a probability of fraud "
            "and it does not prove that a document is fake or genuine."
        )

        risk_result = calculate_risk_score(
            document_type=document_type,
            classification=classification,
            validation_results=validation_results,
            tampering_indicators=tampering_indicators,
            average_ocr_confidence=average_ocr_confidence,
            face_result=face_result,
        )

        risk_score = risk_result["score"]
        risk_level = risk_result["level"]

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric("Risk score", f"{risk_score:.2f} / 100")

        with metric_col2:
            st.metric("Risk level", risk_level)

        if risk_level == "LOW":
            st.success(
                "LOW RISK — few screening concerns were detected. Human review is still recommended."
            )
        elif risk_level == "MEDIUM":
            st.warning(
                "MEDIUM RISK — one or more screening concerns need attention. Manual review is recommended."
            )
        else:
            st.error(
                "HIGH RISK — several screening concerns were detected. Manual review is strongly recommended."
            )

        st.write("**Risk breakdown:**")
        for component_name, component_value in risk_result["breakdown"].items():
            st.write(
                f"- **{component_name}:** {component_value:.2f} points"
            )

        with st.expander("Why this score was assigned"):
            if risk_result["reasons"]:
                for reason in risk_result["reasons"]:
                    st.write(f"- {reason}")
            else:
                st.write("No additional review reasons were recorded.")

        st.caption(
            "Prototype weighting: classification 10 points, validation 30, OCR confidence 20, "
            "image/tampering indicators 20, and face similarity 20. "
            "The weights are designed for explainability and should be evaluated on a larger "
            "synthetic test set before being treated as a production decision rule."
        )

        # ----------------------------------------------
        # Save / report / history
        # ----------------------------------------------
        st.subheader("11. Save Screening & Generate Report")

        report_payload = build_report_payload(
            document_filename=uploaded_file.name,
            document_type=document_type,
            document_type_name=document_type_name,
            average_ocr_confidence=average_ocr_confidence,
            classification=classification,
            validation_results=validation_results,
            tampering_indicators=tampering_indicators,
            face_result=face_result,
            risk_result=risk_result,
        )

        report_json = json.dumps(report_payload, indent=2, ensure_ascii=False, default=str)

        save_col, download_col = st.columns(2)

        with save_col:
            if st.button("Save screening result to SQLite", type="primary"):
                record_id = save_screening_record(
                    document_filename=uploaded_file.name,
                    document_type=document_type,
                    risk_result=risk_result,
                    average_ocr_confidence=average_ocr_confidence,
                    face_result=face_result,
                )
                st.success(
                    f"Screening result saved to SQLite. Record ID: {record_id}"
                )

        with download_col:
            st.download_button(
                "Download JSON screening report",
                data=report_json,
                file_name="sih26188_screening_report.json",
                mime="application/json",
            )

        st.caption(
            f"SQLite database: {DATABASE_PATH.relative_to(BASE_DIR)}"
        )

        recent_records = load_recent_screenings(limit=10)
        if recent_records:
            with st.expander("Recent saved screening records"):
                st.dataframe(
                    recent_records,
                    use_container_width=True,
                    hide_index=True,
                )

        # ----------------------------------------------
        # Final manual-review reminder
        # ----------------------------------------------
        st.subheader("12. Screening Recommendation")
        st.warning(
            "This prototype combines OCR, document rules, image-quality/tampering indicators, "
            "and optional face similarity. It must not automatically declare a document or person "
            "genuine or fraudulent. Human verification remains the final decision step."
        )
