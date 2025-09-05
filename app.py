import os
import zipfile
import shutil
import json
import ast
import re
import tempfile

import gradio as gr
from PIL import Image, ImageDraw
from langchain_core.exceptions import OutputParserException
from openai import OpenAI

from prompts import LONG_SYSTEM_PROMPT, output_parser
from config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL

MAX_SIDE = 1280
ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Конфигурация доступа к vLLM (OpenAI-совместимый сервер)
client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# --- вспомогательные функции (без изменений) ---

def list_assets() -> str:
    files = sorted(os.listdir(ASSETS_DIR))
    if not files:
        return "<i>Каталог пуст</i>"
    items = "\n".join(f"<li>{f}</li>" for f in files)
    return f"<b>Файлы в папке assets({len(files)}):</b><ul>{items}</ul>"

def clear_assets():
    shutil.rmtree(ASSETS_DIR, ignore_errors=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    return list_assets()

def progress_bar(percent: int) -> str:
    return f"""
    <div style='width:100%;background:#ddd;border:1px solid #ccc;height:20px;position:relative'>
        <div style='background:#4caf50;width:{percent}%;height:100%'></div>
        <span style='position:absolute;left:50%;top:0;transform:translate(-50%,0);
                     font-size:12px;font-weight:bold'>{percent}%</span>
    </div>
    """

def handle_zip(zip_file):
    with zipfile.ZipFile(zip_file.name) as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, ASSETS_DIR)
            pct = int(i / total * 100)
            yield progress_bar(pct), gr.update()
    yield "", list_assets()

def normalize_to_json(text: str) -> str:
    txt = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    txt = re.sub(r"^json\s*", "", txt).strip()
    try:
        json.loads(txt)
        return txt
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(txt)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return txt.replace("'", '"')

def plot_bounding_boxes(image: Image.Image, bboxes: list, labels_to_plot=None):
    draw = ImageDraw.Draw(image)
    colors = {
        "Сломана остановка": "red", "Повреждены указатели": "blue",
        "Не окрашена остановка": "green",   "Не восстановлена опора": "yellow",
        "Повалена опора": "orange",         "Не окрашена опора": "pink",
        "Складирование остатков асфальта": "gray", "Отсутствует травяной покров": "black"
    }
    for box in bboxes:
        if not isinstance(box, dict):
            continue
        label = box.get("label")
        coords = box.get("bbox_2d")
        if labels_to_plot and label not in labels_to_plot:
            continue
        if not (isinstance(label, str) and isinstance(coords, (list, tuple)) and len(coords)==4):
            continue
        try:
            x1, y1, x2, y2 = map(int, coords)
        except:
            continue
        x0, x1n = min(x1, x2), max(x1, x2)
        y0, y1n = min(y1, y2), max(y1, y2)
        color = colors.get(label, "purple")
        draw.rectangle(((x0, y0), (x1n, y1n)), outline=color, width=3)
    return image

# --- обновлённая функция обработки изображения ---

def process_image(image_path: str):
    SYSTEM_MSG = {"role": "system", "content": LONG_SYSTEM_PROMPT}
    target = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            {"type": "text", "text": "Проанализируй это изображение: "},
        ],
    }
    messages = [SYSTEM_MSG, target]

    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        temperature=0.7,
    )
    raw = resp.choices[0].message.content if resp.choices else ""

    try:
        parsed = output_parser.parse(raw)
    except OutputParserException:
        norm = normalize_to_json(raw)
        try:
            parsed = output_parser.parse(norm)
        except OutputParserException:
            try:
                data = json.loads(norm)
            except:
                raise
            if isinstance(data, dict) and "analysis" in data:
                data.setdefault("result", [])
                data.setdefault("bounding_boxes", [])
                parsed = data
            else:
                parsed = output_parser.parse(norm)

    parsed.setdefault("result", [])
    parsed.setdefault("bounding_boxes", [])
    return parsed, raw

# --- обновлённая функция analyze_image ---

def analyze_image(img: Image.Image):
    # Подготовка изображения
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)

    # Сохраняем временно
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        path = tmp.name

    parsed, raw = process_image(path)
    os.unlink(path)
    if "error" in parsed:
        return None, parsed["error"]

    annotated = plot_bounding_boxes(img.copy(), parsed["bounding_boxes"])
    return annotated, raw

# --- Gradio UI ---

with gr.Blocks(title="Глаз ППБ") as demo:
    favicon = '<img src="https://cdn-icons-png.flaticon.com/512/3137/3137672.png" width="48px">'
    gr.Markdown(f"<h1><center>{favicon} Глаз ППБ — я тебя вижу</center></h1>")

    with gr.Sidebar():
        gr.Markdown("### Управление архивами")
        files_list = gr.HTML(list_assets)
        gr.Button("Очистить assets", variant="stop").click(
            clear_assets, inputs=None, outputs=files_list
        )

    gr.Markdown("### Проверка одного изображения")

    with gr.Row():
        with gr.Column():
            inp     = gr.Image(type="pil", label="Загрузить изображение")

        with gr.Column():
            out_img = gr.Image(type="pil", label="Результат")

    with gr.Row():
        with gr.Accordion("Анализ", open=False):
            out_raw = gr.Textbox(lines=10, show_label=False)

    with gr.Row():
        btn = gr.Button("Анализировать", variant="primary")


    gr.Markdown("### Загрузить ZIP-архив с фото")
    with gr.Row():
        with gr.Column():
            zip_uploader = gr.File(
                label="Выберите архив",
                file_types=[".zip"],
                type="filepath"
            )
            progress_html = gr.HTML("")

            zip_uploader.upload(
                handle_zip,
                inputs=[zip_uploader],
                outputs=[progress_html, files_list]
            )


    btn.click(
        analyze_image,
        inputs=[inp],
        outputs=[out_img, out_raw]
    )

demo.queue().launch(root_path="/glaz_gkh")
