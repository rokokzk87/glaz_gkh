import os
import zipfile
import shutil
import json
import ast
import re
import tempfile

import torch
import gradio as gr
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from langchain_core.exceptions import OutputParserException
from peft import PeftModel, PeftMixedModel

from prompts import LONG_SYSTEM_PROMPT, output_parser
from few_shots import demo_examples

MAX_SIDE = 1280
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTERS_DIR = "/home/ubuntu/qwen_finetune/adapters/"
ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# 1) Собираем списки адаптеров с опцией "None"
text_adapter_dirs = ["None"] + sorted(
    d for d in os.listdir(ADAPTERS_DIR)
    if os.path.isdir(os.path.join(ADAPTERS_DIR, d)) and d.startswith("text")
)
vision_adapter_dirs = ["None"] + sorted(
    d for d in os.listdir(ADAPTERS_DIR)
    if os.path.isdir(os.path.join(ADAPTERS_DIR, d)) and d.startswith("vision")
)

# 2) По-умолчанию — никакие адаптеры не выбраны
init_text, init_vision = "None", "None"

# 3) Загружаем «чистую» базовую модель
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
base.eval()

# 4) Загружаем PEFT-модель и все адаптеры в неё
#    (для динамического переключения)
first_text = next(d for d in text_adapter_dirs if d != "None")
first_vis  = next(d for d in vision_adapter_dirs if d != "None")
first_text_path   = os.path.join(ADAPTERS_DIR, first_text,   "text_lora")
first_vision_path = os.path.join(ADAPTERS_DIR, first_vis,     "vision_lora")

model = PeftMixedModel.from_pretrained(
    base,
    first_text_path,
    adapter_name=first_text,
    is_trainable=False
)
model.load_adapter(
    first_vision_path,
    adapter_name=first_vis,
    is_trainable=False
)
for name in text_adapter_dirs:
    if name in ("None", first_text):
        continue
    path = os.path.join(ADAPTERS_DIR, name, "text_lora")
    model.load_adapter(path, adapter_name=name, is_trainable=False)

for name in vision_adapter_dirs:
    if name in ("None", first_vis):
        continue
    path = os.path.join(ADAPTERS_DIR, name, "vision_lora")
    model.load_adapter(path, adapter_name=name, is_trainable=False)

model.eval()

# 5) Процессор
processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True)

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

def process_image(model_to_use, device, image_path: str):
    SYSTEM_MSG = {"role": "system", "content": LONG_SYSTEM_PROMPT}
    few_shots = []
    for ex in demo_examples:
        few_shots.extend([
            {"role": "user", "content": [{"type": "image", "image": ex["image"]}]},
            {"role": "assistant", "content": json.dumps(ex["json"], ensure_ascii=False)}
        ])
    target = {
        "role": "user",
        "content": [
            {"type":"image","image":image_path},
            {"type":"text","text":"Проанализируй это изображение: "}
        ]
    }
    messages = [SYSTEM_MSG] + few_shots + [target]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    vision_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=vision_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        gen_ids = model_to_use.generate(
            **inputs,
            do_sample=True,
            temperature=0.25,
            top_p=0.95,
            repetition_penalty=1.1,
            max_new_tokens=2048,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    raw = processor.batch_decode(
        gen_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

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

def analyze_image(img: Image.Image, text_adapter: str, vision_adapter: str):
    # Собираем выбранные адаптеры
    adapters = []
    if text_adapter != "None":
        adapters.append(text_adapter)
    if vision_adapter != "None":
        adapters.append(vision_adapter)

    # Выбираем модель и устройство
    if adapters:
        model.set_adapter(adapters)
        model_to_use = model
        status_html = f"<b>Active adapters:</b> {', '.join(adapters)}"
    else:
        model_to_use = base
        status_html = "<b>Active adapters:</b> None"

    device = next(model_to_use.parameters()).device

    # Подготовка изображения
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)

    # Сохраняем временно
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        path = tmp.name

    parsed, raw = process_image(model_to_use, device, path)
    os.unlink(path)
    if "error" in parsed:
        return None, parsed["error"], raw

    annotated = plot_bounding_boxes(img.copy(), parsed["bounding_boxes"])
    return annotated, raw, status_html

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
        text_dd = gr.Dropdown(label="Text LoRA",   choices=text_adapter_dirs,  value=init_text)
        vis_dd  = gr.Dropdown(label="Vision LoRA", choices=vision_adapter_dirs, value=init_vision)

    with gr.Row():
        status  = gr.HTML(f"<b>Active adapters:</b> {init_text}, {init_vision}")

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
        inputs=[inp, text_dd, vis_dd],
        outputs=[out_img, out_raw, status]
    )

demo.queue().launch(root_path="/glaz_gkh")
