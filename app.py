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
    Google広告 CSVのカラム名に含まれるBOM、空白、引用符などを整理します。
    """
    col = str(col)
    col = col.replace("\ufeff", "")
    col = col.replace('"', "")
    col = col.replace("'", "")
    col = col.strip()
    return col


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrameのカラム名を正規化します。
    """
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def detect_header_row(lines: list[str]) -> int:
    """
    Google広告CSVの上部にレポート名・期間などのメタ情報がある場合、
    実際のヘッダー行を探します。
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

        if hit_count >= 2:
            return i

    return 0


def try_read_csv_text(text: str, sep: str) -> Optional[pd.DataFrame]:
    """
    指定した文字列と区切り文字でCSV/TSV読み込みを試します。
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

        if df.empty or len(df.columns) <= 1:
            return None

        df = df.dropna(axis=1, how="all")

        return df

    except Exception:
        return None


def read_google_ads_csv(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """
    Google広告 CSV/TSVを安全に読み込みます。
    - UTF-8 / CP932 / Shift-JIS / UTF-16
    - カンマ / タブ / セミコロン
    - 上部メタ行スキップ
    - 壊れた行はスキップ
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
        except Exception:
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
    columns = list(df.columns)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    for col in columns:
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
    パフォーマンス = 低 / Low を判定します。
    """
    text = str(value).strip()
    return text in LOW_PERFORMANCE_VALUES


def is_target_asset_type(value) -> bool:
    """
    広告見出し / 説明文 のみ対象にします。
    """
    text = str(value).strip()
    return any(keyword in text for keyword in TARGET_ASSET_TYPE_KEYWORDS)


def filter_low_performance_assets(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    パフォーマンス評価が「低」の広告見出し / 説明文だけ抽出します。
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
    GPTに渡す用に、必要そうなカラムだけを抽出します。
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
        return df.head(120)

    return df[selected_cols].copy()


# ============================================================
# 문자수 제한 검증 함수
# ============================================================

def get_char_limit(asset_type: str) -> int:
    """
    Google広告の文字数制限。
    広告見出し: 30文字
    説明文: 90文字
    """
    asset_type = str(asset_type)

    if "説明文" in asset_type or "Description" in asset_type:
        return 90

    return 30


def count_ad_chars(text: str) -> int:
    """
    日本語1文字を1文字として数えます。
    句読点、記号、スペースも1文字としてカウントします。
    """
    if text is None:
        return 0

    return len(str(text))


def validate_replacement_text(text: str, asset_type: str) -> dict:
    """
    代替案がGoogle広告の文字数制限内か判定します。
    """
    limit = get_char_limit(asset_type)
    length = count_ad_chars(text)

    return {
        "length": length,
        "limit": limit,
        "is_valid": length <= limit,
        "status": "OK" if length <= limit else "文字数超過",
    }


def has_invalid_replacements(result_df: pd.DataFrame) -> bool:
    """
    生成結果に文字数超過が含まれているか確認します。
    """
    if result_df.empty:
        return False

    judge_cols = [col for col in result_df.columns if col.startswith("判定")]

    for col in judge_cols:
        if (result_df[col] == "文字数超過").any():
            return True

    return False


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

文字数制限：
- asset_type が「広告見出し」の場合、replacement_1 / replacement_2 / replacement_3 は必ず30文字以内にしてください。
- asset_type が「説明文」の場合、replacement_1 / replacement_2 / replacement_3 は必ず90文字以内にしてください。
- 日本語1文字を1文字として数えてください。
- 句読点、記号、スペースも1文字として数えてください。
- 30文字または90文字を超える案は絶対に出力しないでください。
- 長くなりそうな場合は、短く自然な広告文に言い換えてください。

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


def build_repair_prompt(
    result_df: pd.DataFrame,
    product_context: str,
    ng_words: str,
) -> str:
    """
    文字数超過が出た場合に、超過案だけ短縮させるためのプロンプトです。
    """
    result_csv = result_df.to_csv(index=False)

    return f"""
あなたはGoogle広告の広告文修正専門家です。

以下の代替案のうち、判定が「文字数超過」になっているものを、
同じ意味と訴求軸を保ったまま文字数制限内に短縮してください。

ルール：
- 広告見出しは30文字以内。
- 説明文は90文字以内。
- OKになっている案も、より自然に短くできる場合は調整してよいです。
- ただし、元の訴求軸から外れないでください。
- 日本語として自然な広告文にしてください。
- 出力はJSONのみ。

補足情報：
{product_context}

NG表現：
{ng_words}

修正対象：
{result_csv}

出力形式：
{{
  "summary": "修正後の要約",
  "items": [
    {{
      "original_asset": "既存アセット",
      "asset_type": "広告見出し or 説明文",
      "issue": "問題推定",
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


def convert_items_to_dataframe(data: dict) -> Tuple[str, pd.DataFrame]:
    """
    OpenAIのJSONレスポンスを表示用DataFrameに変換し、
    Python側で文字数検証を行います。
    """
    rows = []

    for item in data.get("items", []):
        asset_type = item.get("asset_type", "")

        replacement_1 = item.get("replacement_1", "")
        replacement_2 = item.get("replacement_2", "")
        replacement_3 = item.get("replacement_3", "")

        v1 = validate_replacement_text(replacement_1, asset_type)
        v2 = validate_replacement_text(replacement_2, asset_type)
        v3 = validate_replacement_text(replacement_3, asset_type)

        rows.append(
            {
                "既存アセット": item.get("original_asset", ""),
                "タイプ": asset_type,
                "問題推定": item.get("issue", ""),

                "代替案1": replacement_1,
                "文字数1": f"{v1['length']} / {v1['limit']}",
                "判定1": v1["status"],
                "狙い1": item.get("intent_1", ""),

                "代替案2": replacement_2,
                "文字数2": f"{v2['length']} / {v2['limit']}",
                "判定2": v2["status"],
                "狙い2": item.get("intent_2", ""),

                "代替案3": replacement_3,
                "文字数3": f"{v3['length']} / {v3['limit']}",
                "判定3": v3["status"],
                "狙い3": item.get("intent_3", ""),
            }
        )

    result_df = pd.DataFrame(rows)
    summary = data.get("summary", "")

    return summary, result_df


def call_openai_json(
    api_key: str,
    model: str,
    prompt: str,
    image: Optional[Image.Image] = None,
) -> dict:
    """
    OpenAI APIを呼び出してJSONレスポンスを受け取ります。
    """
    client = OpenAI(api_key=api_key)

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

    return extract_json_from_text(response.output_text)


def generate_replacement_ideas(
    api_key: str,
    model: str,
    context_df: pd.DataFrame,
    low_df: pd.DataFrame,
    product_context: str,
    ng_words: str,
    columns: dict,
    image: Optional[Image.Image] = None,
    auto_repair: bool = True,
) -> Tuple[str, pd.DataFrame]:
    """
    OpenAI APIを呼び出し、代替案を生成します。
    文字数超過がある場合は、1回だけ自動修正を試みます。
    """
    prompt = build_prompt(
        context_df=context_df,
        low_df=low_df,
        product_context=product_context,
        ng_words=ng_words,
        columns=columns,
    )

    data = call_openai_json(
        api_key=api_key,
        model=model,
        prompt=prompt,
        image=image,
    )

    summary, result_df = convert_items_to_dataframe(data)

    if auto_repair and has_invalid_replacements(result_df):
        repair_prompt = build_repair_prompt(
            result_df=result_df,
            product_context=product_context,
            ng_words=ng_words,
        )

        repaired_data = call_openai_json(
            api_key=api_key,
            model=model,
            prompt=repair_prompt,
            image=None,
        )

        repaired_summary, repaired_df = convert_items_to_dataframe(repaired_data)

        if not repaired_df.empty:
            summary = repaired_summary or summary
            result_df = repaired_df

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

    auto_repair = st.checkbox(
        "文字数超過を自動修正する",
        value=True,
        help="生成結果が文字数制限を超えた場合、1回だけ自動で短縮修正します。",
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
                    auto_repair=auto_repair,
                )

                st.subheader("全体診断")
                st.write(summary)

                st.subheader("代替案")
                st.dataframe(result_df, use_container_width=True)

                if has_invalid_replacements(result_df):
                    st.warning(
                        "一部の代替案がまだ文字数制限を超えています。"
                        "判定が「文字数超過」の案は手動で短縮してください。"
                    )
                else:
                    st.success("すべての代替案が文字数制限内です。")

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
