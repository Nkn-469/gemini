# 説明欄

## gemnerate_image.pyの一覧

### ライブラリ一覧

- ```torch```【PyTorch（モデルのロードやデバイスの制御）】
- ```diffusers```【Stable Diffusion画像生成パイプライン】
- ```transformers```【Hugging Faceが提供する自然言語処理（NLP）用のモデルライブラリ】
- ```accelerate```【PyTorchでの分散学習やマルチGPU・TPU対応を簡素化】
- ```google-generativeai、genai```【Gemini APIライブラリ】
- ```python-dotenv```【.envファイルにあるAPIキーをPythonプログラム内で簡単に読み込めるようにする】
- ```OS```【OS操作】
- ```sys```【エラーメッセージ表示に使用】
- ```datetime```【ファイル名に使う現在時刻の取得】
- ```argparse```【コマンドライン引数のパース】
- ```re```【正規表現】
- ```json```【JSON形式のデータ読み込みと変換】
- ```dotenv、load_dotenv```【.envファイルからAPIキーなどを読み込む】

#### ライブラリインストール方法
- requirements.txtを入れる
- その後ターミナルでcdコマンドでgenerate_image.pyとrequirements.txtがあるフォルダに移動
  （例）cd path\to\your\gemini-env
- その後コマンドで```pip install -r requirements.txt```をするとすべてのインストールされる

### 操作方法

- ターミナルで```python generate_image.py```で起動する
- ターミナルで何を生成しますか？と出るのでそこに生成したいものをいう
（例）✅ 何を生成しますか？＞白い犬
- 保存先を変更する場合は```--output```または```-O```できる
  （例）```python generate_image.py -o my_creations```
  
### コード説明
- ```def generate_enhanced_prompts(user_prompt: str, gemini_model) -> tuple[str, str]:```【日本語入力対応させるため】
- ```def generate_and_save_image(pipe, prompt: str, negative_prompt: str, save_folder: str):```【Stable Diffusion パイプラインにプロンプトを渡して画像を生成し、保存する。】

------------------------------------------------------------------

## gemini_test.pyの一覧

### ライブラリ一覧
- ```torch```【PyTorch本体。Stable Diffusionの実行するため】
- ```diffusers```【Hugging FaceのStable Diffusionパイプライン】
- ```transformers```【diffuersが内部で使うことがある】
- ```accelerate```【diffuersやtransformersの高速実行する】
- ```googl-generativeai```【Gemini APIを使ってプロンプトを高品質化する】
- ```python-dotenv```【.envファイルからAPIキーなどを読み込むため】
- ```Pillow```【生成された画像を保存するため】
- ```argparse```【CLI引数処理用】
- ```re```【正規表現】
- ```os```【OS操作】
- ```sys```【エラーメッセージ表示に使用】
- ```datetime```【ファイル名に使う現在時刻の取得】
- ```json```【JSON形式のデータ読み込みと変換】

#### ライブラリインストール方法

- ```pip install torch torchvision torchaudio```
- ```pip install diffusers```
- ```pip install transformers```
- ```pip install accelerate```
- ```pip install google-generativeai```
- ```pip install python-dotenv ```
- ```pip install Pillow```

### 概要
- Geminiを使ってプロンプトを高品質化
- Stable Diffusionで画像を生成
- 画像を指定フォルダに保存
- 対話ループで連続的に画像生成が可能

### コード説明

- ```def generate_enhanced_prompts(user_prompt: str, gemini_model) -> tuple[str, str]:```【Gemini APIを使って自然なプロンプトを作る。】
- ```def generate_and_save_image(pipe, prompt: str, negative_prompt: str, save_folder: str):```【Stable Diffusionで画像を生成】

  -------------------------------------------------------------------------

## gemini_Whissperの一覧

### ライブラリ一覧
- ```whisper```【Whisper音声認識ライブラリのインポート】
- ```argparse```【コマンドライン引数処理用】
- ```os```【ファイル操作用】
- ```sys```【標準エラー出力】
- ```tempfile```【一時ファイルの作成用】

#### ライブラリインストール
- ```pip install -U openai-whisper```
- ```pip install moviepy```
- [ffmpeg](https://ffmpeg.org/download.html)を入れる</br>
  ・入れた後```ffmpeg -version```をターミナルで確認して出てればよい

### 操作方法
- **AudioTranscripts**に動画で音としてでてた内容を文字として保存する
- 保存先を変更する場合は```--output```または```-O```できる
- 動画の保存して選ぶやつはmp4っていうフォルダ

