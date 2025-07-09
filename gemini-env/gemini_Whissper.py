import whisper
import argparse
import os
import sys
import tempfile

try:
    from moviepy import VideoFileClip
except ImportError:
    print("❌ エラー: 'moviepy'ライブラリが見つかりません。", file=sys.stderr)
    print("💡 ヒント: 'pip install moviepy' を実行してインストールしてください。", file=sys.stderr)
    sys.exit(1)

def transcribe_media(media_path: str, model_name: str = "base", output_file: str = None):
    """
    Whisperを使用して音声ファイルまたは動画ファイルから音声を日本語に文字起こしします。
    動画の場合は、音声を抽出してから文字起こしを行います。

    Args:
        media_path (str): 文字起こしする音声または動画ファイルのパス。
        model_name (str): 使用するWhisperのモデル名 ("tiny", "base", "small", "medium", "large")。
        output_file (str, optional): 文字起こし結果を保存するファイルパス。
    """
    if not os.path.exists(media_path):
        print(f"❌ エラー: ファイルが見つかりません: {media_path}", file=sys.stderr)
        return

    VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']
    file_extension = os.path.splitext(media_path)[1].lower()
    is_video = file_extension in VIDEO_EXTENSIONS

    audio_path_to_transcribe = media_path
    temp_audio_file = None

    try:
        # 動画ファイルの場合は音声を抽出
        if is_video:
            print(f"📹 動画ファイルが検出されました。音声の抽出を開始します...")
            # 一時ファイル名を安全に生成
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_f:
                temp_audio_file = temp_f.name
            with VideoFileClip(media_path) as video_clip:
                video_clip.audio.write_audiofile(temp_audio_file, logger=None)
                audio_path_to_transcribe = temp_audio_file
                print(f"✅ 音声の抽出が完了しました: {temp_audio_file}")

        # 文字起こし処理
        print(f"🔄 モデル '{model_name}' をロードしています...")
        model = whisper.load_model(model_name)
        print(f"🎤 '{os.path.basename(media_path)}' の文字起こしを開始します... (これには時間がかかる場合があります)")
        result = model.transcribe(audio_path_to_transcribe, language="ja", fp16=False) # fp16=FalseはCPUでの実行を安定させます
        print("\n--- 文字起こし結果 ---")
        print(result["text"])
        print("--------------------")

        # ファイルに出力する
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"✅ 結果をファイルに保存しました: {output_file}")


    except Exception as e:
        print(f"❌ エラーが発生しました: {e}", file=sys.stderr)
        print("💡 ヒント: ffmpegが正しくインストールされているか確認してください。", file=sys.stderr)
    finally:
        # 一時音声ファイルを削除
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            print(f"🗑️ 一時ファイル '{temp_audio_file}' を削除しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisperで音声ファイルや動画ファイルを文字起こしします。")
    parser.add_argument("media_file", help="文字起こしする音声または動画ファイル (例: 'C:\\videos\\my_video.mp4')")
    parser.add_argument("-m", "--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="使用するモデル")
    parser.add_argument("-o", "--output", help="文字起こし結果を保存するテキストファイルのパス (例: transcript.txt)")

    args = parser.parse_args()

    transcribe_media(args.media_file, args.model, args.output)
