import gradio as gr
import sys

from projects.sa2va.gradio.app_utils import\
    process_markdown, show_mask_pred, description, preprocess_video,\
    show_mask_pred_video, image2video_and_save

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import argparse
import os

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

# システムプロンプト（英語）
SYSTEM_PROMPT = """You are an expert AI assistant specialized in analyzing images and videos with advanced segmentation capabilities. 
You can identify, segment, and track objects in both images and videos. When users ask you to identify or locate objects, you can provide precise segmentation masks.
Key capabilities:
- Object detection and segmentation in images
- Object tracking and segmentation across video frames
- Answering questions about visual content with spatial understanding
- Correcting previous responses based on user feedback

Always provide accurate and helpful responses. If a previous answer was incorrect, acknowledge the user's correction and provide the accurate information."""

def parse_args(args):
    parser = argparse.ArgumentParser(description="Sa2VA Demo")
    parser.add_argument('hf_path', help='Sa2VA hf path.')
    return parser.parse_args(args)

def inference(image, video, follow_up, input_str):
    input_image = image
    
    # エラー: 画像と動画の両方が指定されている
    if image is not None and (video is not None and os.path.exists(video)):
        return image, video, "エラー: 画像または動画のいずれか一方のみを入力してください !!!"
    
    # エラー: follow_upでない場合、入力が必要
    if image is None and (video is None or not os.path.exists(video)) and not follow_up:
        return image, video, "エラー: 画像または動画を入力してください !!!"

    # ===== follow_upチェックの状態で処理を分岐 =====
    if not follow_up:
        # ★ follow_upがOFFの場合: 常に新しい会話として扱う（履歴リセット）
        print('Log: 新しい会話を開始（履歴をリセット）')
        global_infos.n_turn = 0
        global_infos.inputs = ''
        
        image = input_image
        global_infos.image_for_show = image
        global_infos.image = image
        video = video
        global_infos.video = video

        if image is not None:
            global_infos.input_type = "image"
        else:
            global_infos.input_type = "video"
        
        # 会話構造: <image>SYSTEM: ... USER: Q1 ASSISTANT:
        text = f"<image>SYSTEM: {SYSTEM_PROMPT}\nUSER: {input_str}\nASSISTANT:"
        
        print(f"Debug - 新しい質問: '{text[:200]}...'")
    else:
        # ★ follow_upがONの場合: 会話履歴を保持して追加質問
        print(f'Log: フォローアップ質問 (ターン {global_infos.n_turn + 1}) - 会話履歴を保持')
        
        # 会話履歴が存在するかチェック
        if global_infos.n_turn == 0 or not global_infos.inputs:
            # 履歴がない場合はエラー
            return global_infos.image_for_show, global_infos.video, "エラー: フォローアップ質問を使用するには、まず最初の質問を送信してください。"
        
        # 会話構造の構築:
        # ターン1後: <image>SYSTEM: ... USER: Q1 ASSISTANT: A1
        # ターン2: <image>SYSTEM: ... USER: Q1 ASSISTANT: A1 USER: Q2 ASSISTANT:
        # ターン3: <image>SYSTEM: ... USER: Q1 ASSISTANT: A1 USER: Q2 ASSISTANT: A2 USER: Q3 ASSISTANT:
        
        conversation_history = global_infos.inputs  # 例: "<image>SYSTEM: ... USER: Q1 ASSISTANT: A1"
        current_question = input_str
        
        # 会話履歴 + 新しい質問
        text = conversation_history + f"\nUSER: {current_question}\nASSISTANT:"
        
        print(f"Debug - 会話履歴長さ: {len(conversation_history)} 文字")
        print(f"Debug - 追加質問: '{current_question}'")
        print(f"Debug - トークン消費予測: ~{len(text.split())} tokens")
        
        image = global_infos.image
        video = global_infos.video

    input_type = global_infos.input_type
    
    # ビデオ処理
    if input_type == "video":
        try:
            if video is None or not os.path.exists(str(video)):
                return global_infos.image_for_show, global_infos.video, "エラー: ビデオファイルが見つかりません"
            
            video = preprocess_video(video, text)
            
            if not video or len(video) == 0:
                return global_infos.image_for_show, global_infos.video, "エラー: ビデオフレームを抽出できませんでした"
        except Exception as e:
            print(f"ビデオ処理エラー: {e}")
            import traceback
            traceback.print_exc()
            return global_infos.image_for_show, global_infos.video, f"ビデオ処理エラー: {str(e)}"

    # デバッグ出力
    print(f"Debug - input_type: {input_type}")
    print(f"Debug - follow_up: {follow_up}")
    print(f"Debug - ターン番号: {global_infos.n_turn}")
    print(f"Debug - テキスト長: {len(text)} 文字")
    print(f"Debug - '<image>'トークン数: {text.count('<image>')}")
    
    # モデル入力準備: past_textは常に空文字列
    if input_type == "image":
        input_dict = {
            'image': image,
            'text': text,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }
    else:
        input_dict = {
            'video': video,
            'text': text,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }

    # モデル実行
    try:
        return_dict = sa2va_model.predict_forward(**input_dict)
    except AssertionError as e:
        print(f"AssertionError during model inference: {e}")
        print(f"Debug info - text length: {len(text)}")
        print(f"Debug info - text: '{text[:300]}...'" if len(text) > 300 else f"Debug info - text: '{text}'")
        import traceback
        traceback.print_exc()
        return global_infos.image_for_show, global_infos.video, "エラー: モデルの処理に失敗しました。"
    except Exception as e:
        print(f"Unexpected error during model inference: {e}")
        import traceback
        traceback.print_exc()
        return global_infos.image_for_show, global_infos.video, f"予期しないエラー: {str(e)}"
    
    # デバッグ情報
    print(f"return_dict keys: {return_dict.keys()}")
    
    # ===== 会話履歴の更新 =====
    prediction = return_dict.get('prediction', '').strip()
    
    # ★ 常に会話履歴を保存（システムプロンプト含む完全な会話構造を維持）
    global_infos.inputs = text + " " + prediction
    print(f"Debug - 会話履歴を更新: {len(global_infos.inputs)} 文字")
    
    # セグメンテーション結果の処理
    if 'prediction_masks' in return_dict.keys() and return_dict['prediction_masks'] and len(
            return_dict['prediction_masks']) != 0:
        if input_type == "image":
            image_mask_show, selected_colors = show_mask_pred(global_infos.image_for_show, return_dict['prediction_masks'])
            video_mask_show = global_infos.video
        else:
            image_mask_show = None
            video_mask_show, selected_colors = show_mask_pred_video(video, return_dict['prediction_masks'])
            video_mask_show = image2video_and_save(video_mask_show, save_path="./ret_video.mp4")
    else:
        image_mask_show = global_infos.image_for_show
        video_mask_show = global_infos.video
        selected_colors = []

    global_infos.n_turn += 1

    predict = process_markdown(prediction, selected_colors)
    return image_mask_show, video_mask_show, predict

def update_follow_up_state(image, video):
    """画像/動画の入力状態とターン数に応じてFollow upボタンの状態を更新"""
    # 最初の質問が送信済みで、画像または動画が存在すれば有効化
    has_input = image is not None or (video is not None and os.path.exists(str(video)) if video else False)
    has_history = global_infos.n_turn > 0
    
    if has_input and has_history:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False, value=False)

def clear_all():
    """全ての状態をクリア"""
    global_infos.n_turn = 0
    global_infos.inputs = ''
    global_infos.image_for_show = None
    global_infos.image = None
    global_infos.video = None
    global_infos.input_type = "image"
    print('Log: All states cleared!')
    return None, None, gr.update(interactive=False, value=False), "", None, None, ""

def init_models(args):
    model_path = args.hf_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    return model, tokenizer

class global_infos:
    inputs = ''
    n_turn = 0
    image_width = 0
    image_height = 0

    image_for_show = None
    image = None
    video = None

    input_type = "image"

if __name__ == "__main__":
    # get parse args and set models
    args = parse_args(sys.argv[1:])

    sa2va_model, tokenizer = init_models(args)

    with gr.Blocks(theme=gr.themes.Soft(), title='Sa2VA') as demo:
        gr.Markdown("# Sa2VA")
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 入力")
                image_input = gr.Image(type="pil", label="画像をアップロード", height=360)
                video_input = gr.Video(sources=["upload", "webcam"], label="MP4動画をアップロード", height=360)
                
                with gr.Row():
                    follow_up_checkbox = gr.Checkbox(
                        label="フォローアップ質問", 
                        interactive=False,
                        value=False,
                        info="最初の質問を送信後に有効化されます"
                    )
                
                text_input = gr.Textbox(
                    lines=2, 
                    placeholder="質問を入力してください...", 
                    label="テキスト指示"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("送信", variant="primary", scale=2)
                    clear_btn = gr.Button("クリア", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("### 出力")
                image_output = gr.Image(type="pil", label="出力画像", height=360)
                video_output = gr.Video(
                    label="出力動画", 
                    show_download_button=True, 
                    format='mp4',
                    height=360
                )
                markdown_output = gr.Markdown()
        
        # 送信後にFollow upボタンの状態を更新
        def inference_and_update(image, video, follow_up, input_str):
            # 推論実行
            img_out, vid_out, text_out = inference(image, video, follow_up, input_str)
            
            # Follow upボタンの状態を更新
            has_input = image is not None or (video is not None and os.path.exists(str(video)) if video else False)
            has_history = global_infos.n_turn > 0
            
            if has_input and has_history:
                follow_up_state = gr.update(interactive=True)
            else:
                follow_up_state = gr.update(interactive=False, value=False)
            
            return img_out, vid_out, text_out, follow_up_state
        
        # 送信ボタンのクリック処理
        submit_btn.click(
            fn=inference_and_update,
            inputs=[image_input, video_input, follow_up_checkbox, text_input],
            outputs=[image_output, video_output, markdown_output, follow_up_checkbox]
        )
        
        # クリアボタンの処理
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                image_input, 
                video_input, 
                follow_up_checkbox, 
                text_input, 
                image_output, 
                video_output, 
                markdown_output
            ]
        )
        
        # Enterキーで送信
        text_input.submit(
            fn=inference_and_update,
            inputs=[image_input, video_input, follow_up_checkbox, text_input],
            outputs=[image_output, video_output, markdown_output, follow_up_checkbox]
        )
    
    demo.queue()
    demo.launch(share=True)