import os



from modules import script_callbacks
import gradio as gr
from scripts.module.anime_face import detect

def daf_tab():
    with gr.Blocks() as main_block:
        with gr.Column():
            output_html = gr.HTML(elem_id=f'output_text')
            # progress_bar = gr.HTML(elem_id=f'progress_bar')
        with gr.Column():
            detect_button = gr.Button(value="Detect!", variant="primary")
            padding = gr.Slider(label="padding", minimum=0, maximum=520, step=1, value=20)
            input_directory = gr.Text(label="Input directory")
            output_directory = gr.Text(label="Output directory")
            with gr.Row():
                chk_detection_results = gr.Checkbox(label="Output Detection Results", value=False)
            
        detect_button.click(
            fn=detect,
            # _js="ProgressUpdate",
            inputs=[input_directory, output_directory,padding,chk_detection_results],
            outputs=[output_html],
            show_progress = True,
        )
            
    return (main_block, "Detect Anime Face", "detect_anime_face"),


script_callbacks.on_ui_tabs(daf_tab)


