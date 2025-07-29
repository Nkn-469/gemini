import whisper
import argparse
import os
import sys
import tempfile

# --- ここから追加 ---
# システム環境変数を変更できない場合、ここにffmpeg.exeのフルパスを設定します。
# 以下のパスはご自身の環境に合わせて書き換えてください。
# パスの先頭に r を付けるのを忘れないでください。(例: r"C:\ffmpeg\bin\ffmpeg.exe")
FFMPEG_PATH = r"C:\Users\mitsuyuki-kurashiki\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# moviepyがffmpegを見つけられるように環境変数を設定
if os.path.exists(FFMPEG_PATH):
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
else:
    # パスが正しくない場合、ユーザーに知らせる
    print(f"⚠️ 警告: 指定されたffmpegのパスが見つかりません: {FFMPEG_PATH}", file=sys.stderr)
    print("💡 ヒント: スクリプト内の FFMPEG_PATH を正しいパスに修正してください。", file=sys.stderr)
# --- ここまで追加 ---

try:
    # moviepy.editor は多くのモジュールをインポートするため、環境によっては不要な依存関係で問題が発生することがあります。
    # 必要な VideoFileClip のみを直接インポートすることで、問題を回避しやすくなります。
    from moviepy.video.io.VideoFileClip import VideoFileClip
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
        # verbose=False を指定することで、Whisperライブラリからの英語の進捗表示を抑制します。
        result = model.transcribe(audio_path_to_transcribe, language="ja", fp16=False, verbose=False) # fp16=FalseはCPUでの実行を安定させます
        print("✅ 文字起こしが完了しました。")

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
    parser.add_argument("media_file", nargs='?', default=None, help="文字起こしする音声または動画ファイル。指定しない場合は、'mp4'フォルダ内のファイルを選択します。 (例: 'my_video.mp4' または 'C:\\videos\\my_video.mp4')")
    parser.add_argument("-m", "--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="使用するモデル")
    parser.add_argument("-o", "--output", help="文字起こし結果を保存するテキストファイルのパス。指定しない場合、'AudioTranscripts'フォルダに自動で保存されます。")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_media_path = None

    # --- 入力ファイルのパスを解決 ---
    if args.media_file:
        # 引数が指定された場合は、従来のパス解決ロジックを使用
        input_media_path = args.media_file
        if not os.path.isabs(input_media_path) and os.path.basename(input_media_path) == input_media_path:
            potential_path = os.path.join(script_dir, "mp4", input_media_path)
            if os.path.exists(potential_path):
                input_media_path = potential_path
                print(f"ℹ️ 'mp4'フォルダからファイル '{os.path.basename(input_media_path)}' を読み込みます。")
    else:
        # 引数が指定されない場合は、ファイル選択モード
        mp4_folder = os.path.join(script_dir, "mp4")
        if not os.path.isdir(mp4_folder):
            print(f"❌ エラー: 'mp4' フォルダが見つかりません。", file=sys.stderr)
            print(f"💡 ヒント: スクリプトと同じ階層に 'mp4' フォルダを作成し、メディアファイルを入れてください。", file=sys.stderr)
            sys.exit(1)

        # サポートするメディア拡張子
        SUPPORTED_MEDIA_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.mp3', '.wav', '.m4a', '.flac']
        media_files = sorted([f for f in os.listdir(mp4_folder) if os.path.splitext(f)[1].lower() in SUPPORTED_MEDIA_EXTENSIONS])

        if not media_files:
            print(f"❌ エラー: 'mp4' フォルダに処理対象のファイルが見つかりません。", file=sys.stderr)
            sys.exit(1)

        print("📂 'mp4' フォルダ内のファイル一覧:")
        for i, filename in enumerate(media_files):
            print(f"  [{i + 1}] {filename}")

        while True:
            try:
                choice = input(f"\n文字起こしするファイルの番号を選択してください (1-{len(media_files)}): ")
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(media_files):
                    selected_file = media_files[choice_index]
                    input_media_path = os.path.join(mp4_folder, selected_file)
                    print(f"✅ '{selected_file}' を選択しました。")
                    break
                else:
                    print(f"⚠️ 1から{len(media_files)}の間の数値を入力してください。", file=sys.stderr)
            except (ValueError, IndexError):
                print("⚠️ 正しい数値を入力してください。", file=sys.stderr)
            except (KeyboardInterrupt, EOFError):
                print("\nキャンセルしました。")
                sys.exit(0)

    # --- 出力ファイルのパスを解決 ---
    output_path = args.output
    if output_path is None:
        # 1. 保存先フォルダのパスを作成
        save_folder = os.path.join(script_dir, "AudioTranscripts")
        os.makedirs(save_folder, exist_ok=True) # フォルダがなければ作成

        # 2. 入力ファイル名から出力ファイル名を生成
        base_filename = os.path.basename(input_media_path) # 解決後のパスからファイル名を取得
        filename_without_ext = os.path.splitext(base_filename)[0]
        output_filename = f"{filename_without_ext}.txt"

        # 3. 最終的な出力パスを結合
        output_path = os.path.join(save_folder, output_filename)
    # -o オプションで出力先が指定されていて、それが相対パスの場合、スクリプトの場所を基準にする
    elif not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)

    transcribe_media(input_media_path, args.model, output_path)
