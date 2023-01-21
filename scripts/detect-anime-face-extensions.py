import os



from modules import script_callbacks
import gradio as gr
from scripts.module.anime_face import detect

def daf_tab():
    with gr.Blocks() as main_block:

        # progress bar
#        with gr.Column():
#             progress_bar = gr.HTML(elem_id=f'progress_bar')

        with gr.Column():
            detect_button = gr.Button(value="Detect!", variant="primary")
            padding = gr.Slider(label="padding", minimum=0, maximum=520, step=1, value=20)
            input_directory = gr.Text(label="Input directory")
            output_directory = gr.Text(label="Output directory")
            debug_output_directory = gr.Text(label="Output Detection Results Directory")

            gr.HTML(value="分からない場合はデフォルトのvalueのままでOK")
            with gr.Row():
                with gr.Column():
                    gr.HTML(value="scaleFactorは値が大きいほど高速化されますが、一部の顔を見落とします。1.05でほとんどの顔を検出できますが、速度は遅くなります。")
                    sclae_factor = gr.Slider(1.0, 1.4, value=1.1,step=0.01,label="scaleFactor")
                with gr.Column():
                    gr.HTML(value="minNeigborsは検出された顔の品質に影響します。値が大きいほど検出数は少なくなりますが、品質は高くなります。3~6が妥当な値です。")
                    min_neighbors = gr.Slider(1, 10, value=5,step=1,label="minNeigbors")
            with gr.Row():
                chk_detection_results = gr.Checkbox(label="Output Detection Results", value=False)
        with gr.Column():
            output_html = gr.HTML(elem_id=f'output_text')
            # progress_bar = gr.HTML(elem_id=f'progress_bar')
            
        detect_button.click(
            fn=detect,
            # _js="ProgressUpdate",
            inputs=[input_directory, output_directory,debug_output_directory,padding,chk_detection_results,sclae_factor,min_neighbors],
            outputs=[output_html],
            show_progress = True,
        )
            
    return (main_block, "Detect Anime Face", "detect_anime_face"),


script_callbacks.on_ui_tabs(daf_tab)


