
# streamlit_app.py
import os
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 0) 페이지/스타일 설정
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기 (스냅샷)", page_icon="🤖")

st.markdown("""
<style>
h1 { color: #1E88E5; text-align: center; font-weight: bold; }
.stFileUploader, .stCameraInput {
  border: 2px dashed #1E88E5; border-radius: 10px; padding: 15px; background-color: #f5fafe;
}
.prediction-box {
  background-color: #E3F2FD; border: 2px solid #1E88E5; border-radius: 10px;
  padding: 25px; text-align: center; margin: 20px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.prediction-box h2 { color: #0D47A1; margin: 0; font-size: 2.0rem; }
.prob-card {
  background-color: #FFFFFF; border-radius: 8px; padding: 15px; margin: 10px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.08); transition: transform 0.2s ease;
}
.prob-card:hover { transform: translateY(-3px); }
.prob-label { font-weight: bold; font-size: 1.05rem; color: #333; }
.prob-bar-bg { background-color: #E0E0E0; border-radius: 5px; width: 100%; height: 22px; overflow: hidden; }
.prob-bar-fg {
  background-color: #4CAF50; height: 100%; border-radius: 5px 0 0 5px; text-align: right;
  padding-right: 8px; color: white; font-weight: bold; line-height: 22px; transition: width 0.5s ease-in-out;
}
.prob-bar-fg.highlight { background-color: #FF6F00; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 카메라 스냅샷/파일 업로드")

# ======================
# 1) 세션 상태 초기화
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None

# ======================
# 2) 모델 로드 (Google Drive)
# ======================
# secrets.toml에 넣어두면 편리합니다. 없으면 기본값 사용
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1t3jtHl0Coivfiq8_sAVFrod-GE_ZvWR8")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    # CPU 강제 로드
    learner = load_learner(output_path, cpu=True)
    return learner

with st.spinner("🤖 AI 모델을 불러오는 중입니다..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 3) 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])

new_bytes = None

with tab_cam:
    st.write("카메라 권한을 허용한 뒤, 스냅샷을 촬영하세요.")
    camera_photo = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if camera_photo is not None:
        new_bytes = camera_photo.getvalue()

with tab_file:
    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
        type=["jpg", "png", "jpeg", "webp", "tiff"]
    )
    if uploaded_file is not None:
        new_bytes = uploaded_file.getvalue()

# 리런에도 유지되도록 세션에 저장
if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 4) 전처리/유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    """EXIF 회전 보정 + RGB 강제."""
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil

# ======================
# 5) 사이드바 옵션(선택)
# ======================
with st.sidebar:
    st.header("설정")
    resize_on = st.toggle("입력 리사이즈 사용", value=False, help="느리다면 켜서 속도를 개선하세요.")
    target_size = st.slider("리사이즈 한 변 길이", min_value=128, max_value=1024, value=384, step=32)
    st.caption("모델 파일은 최초 1회 다운로드 후 캐시됩니다.")
    st.write(f"**모델 파일**: `{MODEL_PATH}`")
    st.write(f"**Drive File ID**: `{FILE_ID}`")

# ======================
# 6) 예측 파이프라인
# ======================
if st.session_state.img_bytes:
    col1, col2 = st.columns([1, 1], vertical_alignment="top")

    # PIL 로드 + 선택적 리사이즈
    try:
        pil_img = load_pil_from_bytes(st.session_state.img_bytes)
        if resize_on:
            # 작은 변 기준 비율 유지 리사이즈(간단 구현: 정사각 리사이즈)
            pil_img = pil_img.resize((target_size, target_size))
    except Exception as e:
        st.exception(e)
        st.stop()

    with col1:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    # fastai 입력: numpy 배열 → PILImage.create
    try:
        fa_img = PILImage.create(np.array(pil_img))
    except Exception as e:
        st.exception(e)
        st.stop()

    with st.spinner("🧠 이미지를 분석 중입니다..."):
        try:
            prediction, pred_idx, probs = learner.predict(fa_img)
        except Exception as e:
            st.exception(e)
            st.stop()

    with col1:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size: 1.0rem; color: #555;">예측 결과:</span>
                <h2>{prediction}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("<h3>상세 예측 확률:</h3>", unsafe_allow_html=True)

        # 파이썬 float로 변환 후 내림차순 정렬
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1],
            reverse=True
        )

        for label, prob in prob_list:
            highlight_class = "highlight" if label == str(prediction) else ""
            prob_percent = prob * 100.0
            st.markdown(
                f"""
                <div class="prob-card">
                    <span class="prob-label">{label}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fg {highlight_class}" style="width: {prob_percent:.4f}%;">
                            {prob_percent:.2f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("카메라에서 스냅샷을 촬영하거나, 파일을 업로드하면 AI가 분석을 시작합니다.")
비공개 댓글 추가…

비공개 댓글은 나와 교사에게만 표시됩니다.
(2학기 10차시) Streamlit을 활용한 인공지능 이미지 분류 웹페이지 만들기
# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1F5RhQzxeztcU7kJpPxG72YlRiTf8Ly0X")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
3번.txt
3번.txt 표시 중입니다.
