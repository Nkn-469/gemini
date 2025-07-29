import torch
from diffusers import StableDiffusionPipeline
import os
import sys
from datetime import datetime
import argparse
import re
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import json
import uuid
import time
from typing import Tuple

def save_image(image, save_folder, prompt):
    """画像を保存する処理を別関数に切り出し"""
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = re.sub(r'[\\/*?:"<>|]', "", prompt)[:40]  # 40文字に制限
    unique_id = uuid.uuid4().hex[:6]  # ユニークIDを生成
    filename = f"{timestamp}_{prompt_slug}_{unique_id}.png"
    save_path = os.path.join(save_folder, filename)
    image.save(save_path)
    print(f"✅ 画像を保存しました: {save_path}")

def generate_enhanced_prompts(user_prompt: str, gemini_model) -> Tuple[str, str]:
    """Geminiでプロンプトを強化"""
    print("🤖 Geminiが最適なプロンプトを考えています...")
    try:
        instruction = f"""あなたは、Stable Diffusionのような画像生成AIのための、世界最高のプロンプトエンジニアです。
ユーザーの簡単な日本語プロンプトを、高品質な画像を生成するための、詳細で具体的な英語のプロンプトに変換してください。

# 指示
- 以下のJSON形式で、ポジティブプロンプトとネガティブプロンプトを生成してください。
- ポジティブプロンプトには、被写体、画風、構図、背景、照明、色使い、雰囲気などを詳細に記述してください。傑作(masterpiece)、最高品質(best quality)などの品質を向上させるキーワードを必ず含めてください。
- ネガティブプロンプトには、生成してほしくない要素（例：低品質、不鮮明、変形、テキスト、署名など）を具体的に記述してください。

# ユーザープロンプト
{user_prompt}

# 出力形式 (JSON)
{{
  "prompt": "ここにポジティブプロンプトを生成",
  "negative_prompt": "ここにネガティブプロンプトを生成"
}}
"""
        response = gemini_model.generate_content(instruction)        
        # Geminiからのレスポンスは .text プロパティにテキストとして格納されます。
        # プロンプトでJSON形式の出力を指示しているので、.text の内容をパースします。
        if response.text:
            # マークダウンのコードブロック(` ```json ... ``` `)を削除
            cleaned_text = re.sub(r'```json\s*(.*?)\s*```', r'\1', response.text, flags=re.DOTALL)
            prompts = json.loads(cleaned_text.strip())

            # 強化されたプロンプトを確認
            print(f"✨ ポジティブプロンプト: {prompts.get('prompt', '')}")
            print(f"🚫 ネガティブプロンプト: {prompts.get('negative_prompt', '')}")            
            return prompts['prompt'], prompts['negative_prompt']
        else:
            print("⚠️ Geminiから適切なプロンプトが取得できませんでした。")
            return f"masterpiece, best quality, photorealistic, {user_prompt}", "deformed, disfigured, ugly, blurry, low quality, low-res, text, watermark, signature, username"

    except Exception as e:
        print(f"⚠️ Geminiでのプロンプト生成に失敗しました: {e}", file=sys.stderr)
        return f"masterpiece, best quality, photorealistic, {user_prompt}", "deformed, disfigured, ugly, blurry, low quality, low-res, text, watermark, signature, username"

def generate_and_save_image(pipe, prompt: str, negative_prompt: str, save_folder: str, guidance_scale: float, num_inference_steps: int):
    """画像を生成し保存する処理"""
    try:
        print("-" * 20)
        print(f"🎨 '{prompt}' の画像を生成しています...")
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        save_image(image, save_folder, prompt)
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}", file=sys.stderr)

def main():
    """スクリプトのメイン処理"""
    load_dotenv()
    parser = argparse.ArgumentParser(description="画像生成ツール")
    parser.add_argument("-o", "--output", type=str, default="img", help="保存先フォルダ")
    args = parser.parse_args()

    save_folder = os.path.abspath(args.output)

    # Stable Diffusionパイプラインの読み込み
    # CPUで実行するため、torch_dtype=torch.float16の指定を削除
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    
    pipe.to("cpu")

    # Gemini API初期化
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    print("画像を生成するプロンプトを入力してください（'exit'で終了）。")
    while True:
        user_prompt = input("ユーザーのプロンプト: ").strip()
        if user_prompt.lower() in ['exit', 'quit']:
            print("終了します。")
            break
        prompt, negative_prompt = generate_enhanced_prompts(user_prompt, gemini_model)
        generate_and_save_image(pipe, prompt, negative_prompt, save_folder, guidance_scale=8.0, num_inference_steps=50)

if __name__ == "__main__":
    main()
