import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
from swapper import process
from restoration import *

port="8080"
server_name="0.0.0.0"


def swap_faces(source_img, target_img, face_restore, background_enhance, face_upsample, upscale, codeformer_fidelity):
    # Convert Gradio images (PIL) to NumPy arrays
    source_img = np.array(source_img)
    target_img = np.array(target_img)
    
    model = "./checkpoints/inswapper_128.onnx"
    result_image = process([source_img], target_img, "-1", "-1", model)
    
    if face_restore:
        # Ensure checkpoint files are downloaded
        check_ckpts()
        
        # Load the Real-ESRGAN upsampler
        upsampler = set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        # Load the CodeFormer face restoration model
        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                         codebook_size=1024,
                                                         n_head=8,
                                                         n_layers=9,
                                                         connect_list=["32", "64", "128", "256"],
                                                        ).to(device)
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        
        # Restore face details
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image, 
                                        background_enhance, 
                                        face_upsample, 
                                        upscale, 
                                        codeformer_fidelity,
                                        upsampler,
                                        codeformer_net,
                                        device)
        result_image = Image.fromarray(result_image)
     
    return result_image



with gr.Blocks(title="Face Swap & Restoration") as demo:
    with gr.Row():
        source_img = gr.Image(type="pil", label="Source Image")
        target_img = gr.Image(type="pil", label="Target Image")
        swapped_img = gr.Image(type="pil", label="Swapped Image")

    with gr.Column():
        face_restore = gr.Checkbox(value=True, label="Face Restore")
        background_enhance = gr.Checkbox(value=True, label="Background Enhance")
        face_upsample = gr.Checkbox(value=True, label="Face Upsample")
        upscale = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Upscale Factor")
        codeformer_fidelity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="CodeFormer Fidelity")
        swap_button = gr.Button("Swap Faces")

    swap_button.click(
        fn=swap_faces,
        inputs=[source_img, target_img, face_restore, background_enhance, face_upsample, upscale, codeformer_fidelity],
        outputs=swapped_img
    )



if __name__ == "__main__":
    demo.launch(favicon_path="data/favicon.ico",server_name=server_name, server_port=int(port))
