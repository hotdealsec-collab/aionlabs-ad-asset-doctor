# app.py

import streamlit as st
import pandas as pd
import base64
import io
import json
from PIL import Image
from openai import OpenAI


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


# -----------------------------
# Helper functions
# -----------------------------

ASSET_TYPE_COLUMNS = ["アセットタイプ", "Asset type", "アセットの種類"]
PERFORMANCE_COLUMNS = ["パフォーマンス", "Performance"]
ASSET_TEXT_COLUMNS = ["アセット", "Asset", "広告アセット"]

TARGET_ASSET_TYPES = ["広告見出し", "Headline", "説明文", "Description"]
LOW_VALUES = ["低", "Low"]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def filter_low_performance_assets(df: pd.DataFrame) -> pd.DataFrame:
    asset_type_col = find_column(df, ASSET_TYPE_COLUMNS)
    performance_col = find_column(df, PERFORMANCE_COLUMNS)
    asset_text_col = find_column(df, ASSET_TEXT_COLUMNS)

    if not performance_col or not asset_text_col:
        return pd.DataFrame()

    filtered = df[df[performance_col].astype(str).isin(LOW_VALUES)].copy()

    if asset_type_col:
        filtered = filtered[
            filtered[asset_type_col].astype(str).apply(
                lambda x: any(t in x for t in TARGET_ASSET_TYPES)
            )
        ]

    return filtered


def image_to_base64(image):
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_prompt(full_df, low_df, product_context, ng_words):
    full_context = full_df.head(100).to_csv(index=False)
    low_context = low_df.to_csv(index=False)

    return f"""
あなたはGoogle広告の広告アセット改善専門家です。

目的：
Google広告の広告アセットCSVを分析し、パフォーマンス評価が「低」の広告見出し・説明文に対して、
同じ広告グループ内の高評価アセットの傾向を参考にしながら代替案を作成してください。

重要：
- 低評価アセットだけを単体で見ないでください。
- CSV全体の文脈、最良・良のアセット、クリック率、表示回数、CV、CPAを参考にしてください。
- 広告見出しと説明文を区別してください。
- 既存の高評価アセットと完全に重複しないようにしてください。
- 誇張表現、未確認のNo.1表現、断定しすぎる表現は避けてください。
- 日本語のGoogle広告で使いやすい自然な表現にしてください。
- 出力はJSONのみとしてください。

補足情報：
{product_context}

NG表現：
{ng_words}

CSV全体：
{full_context}

低評価アセット：
{low_context}

出力形式：
{{
  "summary": "全体診断の要約",
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
    api_key,
    model,
    full_df,
    low_df,
    product_context,
    ng_words,
    image=None
):
    client = OpenAI(api_key=api_key)

    prompt = build_prompt(
        full_df=full_df,
        low_df=low_df,
        product_context=product_context,
        ng_words=ng_words
    )

    content = [{"type": "input_text", "text": prompt}]

    image_base64 = image_to_base64(image)
    if image_base64:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{image_base64}"
        })

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    text = response.output_text
    data = json.loads(text)

    rows = []

    for item in data.get("items", []):
        rows.append({
            "既存アセット": item.get("original_asset"),
            "タイプ": item.get("asset_type"),
            "問題推定": item.get("issue"),
            "代替案1": item.get("replacement_1"),
            "狙い1": item.get("intent_1"),
            "代替案2": item.get("replacement_2"),
            "狙い2": item.get("intent_2"),
            "代替案3": item.get("replacement_3"),
            "狙い3": item.get("intent_3"),
        })

    return data.get("summary", ""), pd.DataFrame(rows)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("API設定")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="API Keyは保存されません。実行時のみ使用します。"
    )

    model = st.selectbox(
        "Model",
        ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"],
        index=0
    )

    st.header("補足情報")

    product_context = st.text_area(
        "商材・広告グループ・訴求軸",
        placeholder="例：ピッコマのマンガ作品広告。無料訴求、作品名訴求、今すぐ読める訴求を重視。"
    )

    ng_words = st.text_area(
        "NG表現",
        placeholder="例：No.1、必ず、絶対、公式確認できない誇張表現"
    )


# -----------------------------
# Main UI
# -----------------------------

uploaded_csv = st.file_uploader(
    "Google広告の広告アセットCSVをアップロードしてください",
    type=["csv"]
)

uploaded_image = st.file_uploader(
    "広告アセット画面のスクリーンショットをアップロードしてください 任意",
    type=["png", "jpg", "jpeg"]
)

if uploaded_csv:
    try:
        df = pd.read_csv(uploaded_csv)
    except UnicodeDecodeError:
        uploaded_csv.seek(0)
        df = pd.read_csv(uploaded_csv, encoding="cp932")

    st.subheader("アップロードCSVプレビュー")
    st.dataframe(df.head(30), use_container_width=True)

    low_assets = filter_low_performance_assets(df)

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

    if st.button("代替案を生成する"):
        if not api_key:
            st.error("OpenAI API Keyを入力してください。")
        elif low_assets.empty:
            st.warning("低評価アセットがないため、代替案を生成できません。")
        else:
            with st.spinner("代替案を生成中です..."):
                try:
                    summary, result_df = generate_replacement_ideas(
                        api_key=api_key,
                        model=model,
                        full_df=df,
                        low_df=low_assets,
                        product_context=product_context,
                        ng_words=ng_words,
                        image=image
                    )

                    st.subheader("全体診断")
                    st.write(summary)

                    st.subheader("代替案")
                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="CSVでダウンロード",
                        data=csv,
                        file_name="ad_asset_recommendations.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error("代替案の生成中にエラーが発生しました。")
                    st.exception(e)
