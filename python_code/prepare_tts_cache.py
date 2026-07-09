import os
import sys
import shutil
import time

SPACEMIT_NLP_PATH = "/home/hnu/spacemit-demo/examples/NLP"
CACHE_DIR = "/home/hnu/elder_ai/tts_cache"

sys.path.insert(0, SPACEMIT_NLP_PATH)

from spacemit_tts import TTSModel


TEXTS = {
    "startup": "微信辅助已启动。",
    "home": "微信首页。",
    "wechat_chat": "聊天页面。",
    "wechat_select": "功能菜单。",
    "wechat_videochat": "视频通话页面。",
    "exit": "程序已退出。"
}


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("正在加载官方 TTS 模型，请稍候...")
    tts_model = TTSModel()
    print("TTS 模型加载完成")

    for name, text in TEXTS.items():
        target_path = os.path.join(CACHE_DIR, f"{name}.wav")

        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            print(f"已存在，跳过: {target_path}")
            continue

        print("=" * 60)
        print(f"正在生成: {name} -> {text}")

        tmp_path = None
        start = time.time()

        try:
            tmp_path = tts_model.ort_predict(text)
            shutil.copyfile(tmp_path, target_path)

            print(f"生成完成: {target_path}")
            print(f"耗时: {time.time() - start:.2f} 秒")

        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    print("=" * 60)
    print("全部语音缓存生成完成")
    print(f"缓存目录: {CACHE_DIR}")


if __name__ == "__main__":
    main()
