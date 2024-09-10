import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from conversation import Chat
import torch
torch.cuda.empty_cache()
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# videochat
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat import VideoChat


# ========================================
#             Model Initialization
# ========================================
def init_model():
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    chat = Chat(model)
    print('Initialization Finished')
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='请先上传您的视频', interactive=False),gr.update(value="上传 & 开始聊天", interactive=True), chat_state, img_list


def upload_img(gr_img, gr_video, chat_state):
    print(gr_img, gr_video)
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True), chat_state, None
    if gr_video: 
        llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, 8)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='输入并回车'), gr.update(value="开始聊天", interactive=False), chat_state, img_list
    if gr_img:
        llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='输入并回车'), gr.update(value="开始聊天", interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='输入不能为空!'), chatbot, chat_state
    chat_state =  chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list):
    llm_message,llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=1, temperature=0.1)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

with gr.Blocks(title="InternVideo-VideoChat!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:

    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                with gr.Tab("视频", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload").style(height=400)
                with gr.Tab("图像", elem_id='image_tab'):
                    up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload").style(height=400)
            upload_button = gr.Button(value="上传 & 开始聊天", interactive=True, variant="primary")
            clear = gr.Button("重置")
        
        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='请先上传您的视频', interactive=False).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭发送")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄清除")     
    
    chat = init_model()
    upload_button.click(upload_img, [up_image, up_video, chat_state], [up_image, up_video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list], [chatbot, chat_state, img_list]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list], [chatbot, chat_state, img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_image, up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)
# demo.launch(server_name="0.0.0.0", server_port=10034, enable_queue=True)
