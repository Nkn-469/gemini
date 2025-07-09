import torch
from diffusers import StableDiffusionPipeline
import os
import sys
from datetime import datetime
import argparse
import re
import google.generativeai as genai
from dotenv import load_dotenv
import json

def generate_enhanced_prompts(user_prompt: str, gemini_model) -> tuple[str, str]:
    """
    Geminiを使用して、ユーザーの簡単なプロンプトを詳細な英語のプロンプトとネガティブプロンプトに変換します。

    Args:
        user_prompt (str): ユーザーが入力したプロンプト。
        gemini_model: 初期化済みのGeminiモデル。

    Returns:
        tuple[str, str]: (高品質化されたプロンプト, ネガティブプロンプト)
    """
    print("🤖 Geminiが最適なプロンプトを考えています...")
    try:
        instruction = f"""You are an expert prompt engineer for a photorealistic image generation AI. Your task is to take the user's core subject and enhance it into a detailed, high-quality English prompt.

**Crucially, the final prompt must be a direct and faithful representation of the user's core subject.** Do not add unrelated elements or change the main subject. For example, if the user asks for "sunflower", create a prompt for a beautiful sunflower, not a painting of a sunflower by Van Gogh unless they specifically ask for it.

Also, provide a corresponding negative prompt to avoid common image generation issues.

User's Core Subject: 「{user_prompt}」

Provide the output in a JSON format with two keys: "prompt" and "negative_prompt".
Example for "sunflower": {{"prompt": "A vibrant, photorealistic sunflower in full bloom, facing the sun, with detailed yellow petals and a dark, seed-filled center. 4K, high detail, sharp focus.", "negative_prompt": "cartoon, painting, ugly, deformed, blurry, low quality, multiple flowers"}}
"""
        response = gemini_model.generate_content(instruction)
        # AIが返す可能性のあるマークダウンコードブロックを削除して安定性を向上
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        prompts = json.loads(cleaned_text)
        print(f"✨ ポジティブプロンプト: {prompts['prompt']}")
        print(f"🚫 ネガティブプロンプト: {prompts['negative_prompt']}")
        return prompts['prompt'], prompts['negative_prompt']
    except Exception as e:
        print(f"⚠️ Geminiでのプロンプト生成に失敗しました: {e}", file=sys.stderr)
        print("元のプロンプトで画像を生成します。")
        # フォールバック時も、少しでも良い結果が出るようにキーワードを追加
        return f"photorealistic, high quality, {user_prompt}", "ugly, deformed, blurry, low quality, cartoon, anime"

def generate_and_save_image(pipe, prompt: str, negative_prompt: str, save_folder: str):
    """
    ロード済みのStable Diffusionパイプラインを使用して画像を生成し、指定されたフォルダに保存します。

    Args:
        pipe: ロード済みのStableDiffusionPipelineオブジェクト。
        prompt (str): 画像生成のためのプロンプト。
        negative_prompt (str): 生成してほしくない要素を指定するプロンプト。
        save_folder (str): 画像を保存するフォルダのパス。
    """
    try:
        print(f"🚫 ネガティブプロンプト: '{negative_prompt}'")
        # --- 1. 画像生成 ---
        print(f"🎨 '{prompt}' の画像を生成しています...")
        # `pipe`オブジェクトにプロンプトを渡すだけで画像が生成されます。
        # 生成された画像はPIL.Imageオブジェクトです。
        image = pipe(prompt, negative_prompt=negative_prompt).images[0]

        # --- 2. ユニークなファイル名を生成して画像を保存 ---
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # プロンプトをファイル名として使えるようにサニタイズ
        prompt_slug = re.sub(r'[\\/*?:"<>|]', "", prompt)[:50] # ファイル名に使えない文字を削除し、長さを制限
        filename = f"{timestamp}_{prompt_slug}.png"
        save_path = os.path.join(save_folder, filename)

        print(f"💾 画像を保存中... -> {save_path}")
        image.save(save_path)

        print(f"✅ 画像を保存しました: {save_path}")

    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}", file=sys.stderr)

def main():
    """スクリプトのメイン処理"""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Stable Diffusionを使って対話形式で画像を生成し、保存します。",
        formatter_class=argparse.RawTextHelpFormatter # ヘルプの改行を維持
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="img",
        help="画像を保存するフォルダのパス。 (デフォルト: img)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="使用するStable DiffusionモデルのID。 (例: 'stabilityai/stable-diffusion-xl-base-1.0')"
    )
    args = parser.parse_args()

    try:
        # --- Geminiモデルのロード処理 ---
        gemini_model = None
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-1.0-pro')
                print("✅ Geminiプロンプトエンジンが有効です。")
            except Exception as e:
                print(f"⚠️ Geminiの初期化に失敗しました: {e}", file=sys.stderr)
                print("プロンプトの自動生成は無効になります。")
        else:
            print("ℹ️ GOOGLE_API_KEYが設定されていません。プロンプトの自動生成は無効です。")

        # --- モデルのロード処理 (一度だけ実行) ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ℹ️ 使用デバイス: {device}")
        if device == "cpu":
            print("⚠️ 警告: CPUで実行しています。画像の生成には数分かかる場合があります。")

        print(f"🔄 モデル '{args.model_id}' をロードしています... (初回は時間がかかります)")
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print("✅ モデルの準備が完了しました。")

        # --- 対話ループ ---
        print("\n🎨 画像生成を開始します。プロンプトを入力してください。")
        print("（終了するには 'quit' または 'exit' と入力）")
        while True:
            user_prompt = input("✅ 何を生成しますか？ > ").strip()
            if user_prompt.lower() in ["quit", "exit"]:
                print("👋 終了します。")
                break
            if user_prompt:
                # Geminiが使えない場合や失敗した場合の、堅実なデフォルト値を設定
                prompt = f"photorealistic, high quality, {user_prompt}"
                negative_prompt = "ugly, deformed, blurry, low quality, cartoon, anime, text, watermark"
                # Geminiが利用可能なら、プロンプトを高品質化する
                if gemini_model:
                    prompt, negative_prompt = generate_enhanced_prompts(user_prompt, gemini_model)
                generate_and_save_image(pipe, prompt, negative_prompt, args.output)
                print("-" * 20) # 区切り線
    except Exception as e:
        print(f"❌ 致命的なエラーが発生しました: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
