import base64
import io
import json
import re
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI


# ============================================================
# Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="Ad Asset Doctor",
    page_icon="🩺",
    layout="wide"
)

st.title("Ad Asset Doctor｜Google広告アセット診断AI")

st.markdown(
    """
Google広告の広告アセットCSVとスクリーンショットをもとに、  
パフォーマンス評価が「低」の広告見出し・説明文に対して代替案を生成します。
"""
)


# ============================================================
# 컬럼 후보 정의
# ============================================================

ASSET_TEXT_COLUMNS = [
    "アセット",
    "Asset",
    "広告アセット",
    "Asset text",
    "アセット テキスト",
]

ASSET_TYPE_COLUMNS = [
    "アセットタイプ",
    "アセット タイプ",
    "Asset type",
    "アセットの種類",
]

PERFORMANCE_COLUMNS = [
    "パフォーマンス",
    "Performance",
    "パフォーマンス評価",
]

CLICK_COLUMNS = [
    "クリック数",
    "Clicks",
]

CTR_COLUMNS = [
    "クリック率",
    "CTR",
]

IMPRESSION_COLUMNS = [
    "表示回数",
    "Impressions",
]

COST_COLUMNS = [
    "費用",
    "Cost",
]

CONVERSION_COLUMNS = [
    "コンバージョン",
    "Conversions",
]

CPA_COLUMNS = [
    "コンバージョン単価",
    "Cost / conv.",
    "Cost / conversion",
]

TARGET_ASSET_TYPE_KEYWORDS = [
    "広告見出し",
    "Headline",
    "説明文",
    "Description",
]

LOW_PERFORMANCE_VALUES = [
    "低",
    "Low",
]


# ============================================================
# CSV 읽기 관련 함수
# ============================================================

def clean_column_name(col: str) -> str:
    """
    Google広告 CSV의 컬럼명에 포함될 수 있는 공백, BOM, 따옴표 등을 정리합니다.
    """
    col = str(col)
    col = col.replace("\ufeff", "")
    col = col.replace('"', "")
    col = col.replace("'", "")
    col = col.strip()
    return col


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 컬럼명을 정리합니다.
    """
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def detect_header_row(lines: list[str]) -> int:
    """
    Google広告 CSV 상단에 메타 정보가 붙는 경우가 있어,
    실제 헤더 행을 찾아냅니다.
    """
    header_keywords = [
        "アセット",
        "Asset",
        "パフォーマンス",
        "Performance",
        "クリック",
        "Clicks",
        "表示回数",
        "Impressions",
    ]

    for i, line in enumerate(lines):
        hit_count = sum(1 for keyword in header_keywords if keyword in line)

        # 광고 에셋 리포트 헤더일 가능성이 높은 행
        if hit_count >= 2:
            return i

    return 0


def try_read_csv_text(text: str, sep: str) -> Optional[pd.DataFrame]:
    """
    주어진 text와 separator로 pandas read_csv를 시도합니다.
    """
    lines = text.splitlines()
    if not lines:
        return None

    header_row = detect_header_row(lines)
    cleaned_text = "\n".join(lines[header_row:])

    if not cleaned_text.strip():
        return None

    try:
        df = pd.read_csv(
            io.StringIO(cleaned_text),
            sep=sep,
            engine="python",
            on_bad_lines="skip",
        )

        df = normalize_dataframe_columns(df)

        # 1열만 읽힌 경우는 separator 오판 가능성이 높음
        if df.empty or len(df.columns) <= 1:
            return None

        # 완전히 빈 컬럼 제거
        df = df.dropna(axis=1, how="all")

        return df

    except Exception:
        return None


def read_google_ads_csv(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """
    Google広告 CSV/TSV를 최대한 안전하게 읽습니다.
    - UTF-8, CP932, Shift-JIS, UTF-16
    - comma, tab, semicolon
    - 상단 메타 행 스킵
    - bad line skip
    """
    raw = uploaded_file.getvalue()

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp932",
        "shift_jis",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    ]

    separators = [
        ",",
        "\t",
        ";",
    ]

    tried = []

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except Exception as e:
            tried.append(f"{encoding}: decode failed")
            continue

        for sep in separators:
            df = try_read_csv_text(text, sep)

            if df is not None:
                meta = {
                    "encoding": encoding,
                    "separator": "\\t" if sep == "\t" else sep,
                    "rows": len(df),
                    "columns": len(df.columns),
                }
                return df, meta

            tried.append(f"{encoding} / sep={repr(sep)}: parse failed")

    raise ValueError(
        "CSVを読み込めませんでした。Google広告からCSVまたはTSVで再ダウンロードしてください。\n"
        + "\n".join(tried[-10:])
    )


# ============================================================
# 컬럼 탐색 / 필터링 함수
# ============================================================

def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    후보 컬럼명 중 실제 DataFrame에 존재하는 컬럼을 찾습니다.
    완전일치 우선, 그 다음 부분일치.
    """
    normalized_columns = list(df.columns)

    # 완전일치
    for candidate in candidates:
        if candidate in normalized_columns:
            return candidate

    # 부분일치
    for col in normalized_columns:
        for candidate in candidates:
            if candidate.lower() in col.lower():
                return col

    return None


def get_required_columns(df: pd.DataFrame) -> dict:
    """
    앱에서 필요한 주요 컬럼을 자동 탐색합니다.
    """
    return {
        "asset_text": find_column(df, ASSET_TEXT_COLUMNS),
        "asset_type": find_column(df, ASSET_TYPE_COLUMNS),
        "performance": find_column(df, PERFORMANCE_COLUMNS),
        "clicks": find_column(df, CLICK_COLUMNS),
        "ctr": find_column(df, CTR_COLUMNS),
        "impressions": find_column(df, IMPRESSION_COLUMNS),
        "cost": find_column(df, COST_COLUMNS),
        "conversions": find_column(df, CONVERSION_COLUMNS),
        "cpa": find_column(df, CPA_COLUMNS),
    }


def is_low_performance(value) -> bool:
    """
    パフォーマンス = 低 / Low を判定します.
    """
    text = str(value).strip()
    return text in LOW_PERFORMANCE_VALUES


def is_target_asset_type(value) -> bool:
    """
    広告見出し / 説明文 のみ対象にします.
    """
    text = str(value).strip()
    return any(keyword in text for keyword in TARGET_ASSET_TYPE_KEYWORDS)


def filter_low_performance_assets(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    パフォーマンス評価が「低」の広告見出し / 説明文だけ抽出します.
    """
    performance_col = columns.get("performance")
    asset_text_col = columns.get("asset_text")
    asset_type_col = columns.get("asset_type")

    if not performance_col or not asset_text_col:
        return pd.DataFrame()

    filtered = df[df[performance_col].apply(is_low_performance)].copy()

    if asset_type_col:
        filtered = filtered[filtered[asset_type_col].apply(is_target_asset_type)].copy()

    return filtered


def make_context_dataframe(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    GPTに渡す用に必要そうなカラムだけを抽出します。
    컬럼이 없으면 있는 것만 사용합니다.
    """
    selected_cols = []

    for key in [
        "asset_text",
        "asset_type",
        "performance",
        "clicks",
        "ctr",
        "impressions",
        "cost",
        "conversions",
        "cpa",
    ]:
        col = columns.get(key)
        if col and col in df.columns and col not in selected_cols:
            selected_cols.append(col)

    if not selected_cols:
        return df.head(100)

    return df[selected_cols].copy()


# ============================================================
# 이미지 / JSON / OpenAI 관련 함수
# ============================================================

def image_to_base64(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_json_from_text(text: str) -> dict:
    """
    모델 응답에서 JSON만 추출합니다.
    혹시 ```json fence가 붙어도 처리합니다.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 응답 중간에 JSON object가 섞인 경우 대비
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("OpenAI response could not be parsed as JSON.")


def build_prompt(
    context_df: pd.DataFrame,
    low_df: pd.DataFrame,
    product_context: str,
    ng_words: str,
    columns: dict,
) -> str:
    """
    GPT에게 전달할 프롬프트를 구성합니다.
    """
    full_context = context_df.head(120).to_csv(index=False)
    low_context = low_df.to_csv(index=False)

    asset_text_col = columns.get("asset_text")
    asset_type_col = columns.get("asset_type")
    performance_col = columns.get("performance")

    return f"""
あなたはGoogle広告の広告アセット改善専門家です。

目的：
Google広告の広告アセットCSVを分析し、パフォーマンス評価が「低」の広告見出し・説明文に対して、
同じ広告グループ内の高評価アセットの傾向を参考にしながら代替案を作成してください。

重要ルール：
- 低評価アセットだけを単体で見ないでください。
- CSV全体の文脈を見て、最良・良のアセットの表現傾向を参考にしてください。
- 広告見出しと説明文を区別してください。
- 既存の高評価アセットと完全に重複しないようにしてください。
- 誇張表現、未確認のNo.1表現、断定しすぎる表現は避けてください。
- 日本語のGoogle広告で使いやすい自然な表現にしてください。
- クリック率だけでなく、CV意図も考慮してください。
- 出力はJSONのみとしてください。

認識した主要カラム：
- アセット本文: {asset_text_col}
- アセットタイプ: {asset_type_col}
- パフォーマンス: {performance_col}

補足情報：
{product_context}

NG表現：
{ng_words}

CSV全体の文脈：
{full_context}

低評価アセット：
{low_context}

出力形式：
{{
  "summary": "広告グループ全体の診断要約",
  "items": [
    {{
      "original_asset": "既存アセット",
      "asset_type": "広告見出し or 説明文",
      "issue": "低評価の理由推定",
      "replacement_1": "代替案1",
      "intent_1": "狙い1",
      "replacement_2": "代替案2",
      "intent_2": "狙い2",
      "replacement_3": "代替案3",
      "intent_3": "狙い3"
    }}
  ]
}}
"""


def generate_replacement_ideas(
    api_key: str,
    model: str,
    context_df: pd.DataFrame,
    low_df: pd.DataFrame,
    product_context: str,
    ng_words: str,
    columns: dict,
    image: Optional[Image.Image] = None,
) -> Tuple[str, pd.DataFrame]:
    """
    OpenAI API를 호출하여 대체안을 생성합니다.
    """
    client = OpenAI(api_key=api_key)

    prompt = build_prompt(
        context_df=context_df,
        low_df=low_df,
        product_context=product_context,
        ng_words=ng_words,
        columns=columns,
    )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    image_base64 = image_to_base64(image)
    if image_base64:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_base64}",
            }
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    data = extract_json_from_text(response.output_text)

    rows = []
    for item in data.get("items", []):
        rows.append(
            {
                "既存アセット": item.get("original_asset", ""),
                "タイプ": item.get("asset_type", ""),
                "問題推定": item.get("issue", ""),
                "代替案1": item.get("replacement_1", ""),
                "狙い1": item.get("intent_1", ""),
                "代替案2": item.get("replacement_2", ""),
                "狙い2": item.get("intent_2", ""),
                "代替案3": item.get("replacement_3", ""),
                "狙い3": item.get("intent_3", ""),
            }
        )

    result_df = pd.DataFrame(rows)
    summary = data.get("summary", "")

    return summary, result_df


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("API設定")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="API Keyは保存されません。実行時のみ使用します。",
    )

    model = st.text_input(
        "Model",
        value="gpt-4.1-mini",
        help="利用可能なOpenAIモデル名を入力してください。",
    )

    st.header("補足情報")

    product_context = st.text_area(
        "商材・広告グループ・訴求軸",
        placeholder="例：ピッコマのマンガ作品広告。無料訴求、作品名訴求、今すぐ読める訴求を重視。",
        height=120,
    )

    ng_words = st.text_area(
        "NG表現",
        placeholder="例：No.1、必ず、絶対、公式確認できない誇張表現",
        height=100,
    )


# ============================================================
# Main UI
# ============================================================

uploaded_csv = st.file_uploader(
    "Google広告の広告アセットCSV / TSVをアップロードしてください",
    type=["csv", "tsv", "txt"],
)

uploaded_image = st.file_uploader(
    "広告アセット画面のスクリーンショットをアップロードしてください 任意",
    type=["png", "jpg", "jpeg"],
)

if uploaded_csv:
    try:
        df, read_meta = read_google_ads_csv(uploaded_csv)

    except Exception as e:
        st.error("CSVの読み込みに失敗しました。")
        st.write(
            """
Google広告から出力したファイルに、上部メタ情報・タブ区切り・Shift-JIS/UTF-16などが含まれている可能性があります。  
再度CSVまたはTSV形式でダウンロードして試してください。
"""
        )
        st.exception(e)
        st.stop()

    st.success(
        f"CSVを読み込みました。encoding={read_meta['encoding']} / sep={read_meta['separator']} / "
        f"rows={read_meta['rows']} / columns={read_meta['columns']}"
    )

    columns = get_required_columns(df)

    with st.expander("読み込んだカラムを確認する", expanded=False):
        st.write(list(df.columns))

    with st.expander("認識した主要カラム", expanded=True):
        st.json(columns)

    missing_required = []

    if not columns.get("asset_text"):
        missing_required.append("アセット")
    if not columns.get("performance"):
        missing_required.append("パフォーマンス")

    if missing_required:
        st.error(
            "必要なカラムを認識できませんでした: "
            + ", ".join(missing_required)
            + "。Google広告の広告アセットレポートCSVを確認してください。"
        )
        st.stop()

    st.subheader("アップロードCSVプレビュー")
    st.dataframe(df.head(30), use_container_width=True)

    context_df = make_context_dataframe(df, columns)
    low_assets = filter_low_performance_assets(df, columns)

    st.subheader("低評価アセット候補")

    if low_assets.empty:
        st.info("パフォーマンス評価が「低」の広告見出し・説明文が見つかりませんでした。")
    else:
        st.dataframe(low_assets, use_container_width=True)

    image = None

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.subheader("アップロード画像")
        st.image(image, use_container_width=True)

    st.divider()

    if st.button("代替案を生成する", type="primary"):
        if not api_key:
            st.error("OpenAI API Keyを入力してください。")
            st.stop()

        if low_assets.empty:
            st.warning("低評価アセットがないため、代替案を生成できません。")
            st.stop()

        with st.spinner("代替案を生成中です..."):
            try:
                summary, result_df = generate_replacement_ideas(
                    api_key=api_key,
                    model=model,
                    context_df=context_df,
                    low_df=low_assets,
                    product_context=product_context,
                    ng_words=ng_words,
                    columns=columns,
                    image=image,
                )

                st.subheader("全体診断")
                st.write(summary)

                st.subheader("代替案")
                st.dataframe(result_df, use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="CSVでダウンロード",
                    data=csv_data,
                    file_name="ad_asset_recommendations.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error("代替案の生成中にエラーが発生しました。")
                st.exception(e)
