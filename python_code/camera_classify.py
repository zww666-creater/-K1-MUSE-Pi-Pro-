import cv2
import numpy as np
import onnxruntime as ort

# 类别（与训练时一致）
classes = ['home', 'wechat_chat', 'wechat_select', 'wechat_videochat']

# 加载 ONNX 模型
session = ort.InferenceSession("models/wechat_page.onnx")

# 预处理函数
def preprocess(frame):
    # BGR 转 RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize 到 224x224
    resized = cv2.resize(rgb, (224, 224))
    # 归一化到 [0, 1]
    img = resized.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    # 添加 batch 维度
    img = np.expand_dims(img, axis=0)
    return img

# 推理函数
def predict(frame):
    input_tensor = preprocess(frame)
    outputs = session.run(["output"], {"input": input_tensor})
    pred_idx = np.argmax(outputs[0][0])
    confidence = np.max(outputs[0][0])
    return classes[pred_idx], confidence

# 打开摄像头
cap = cv2.VideoCapture("/dev/wechat_camera")

if not cap.isOpened():
    print("❌ 无法打开摄像头，尝试 /dev/video1")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ 无法打开摄像头，尝试 /dev/video2")
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ 摄像头打开失败，请检查连接")
    exit()

print("✅ 摄像头已打开")
print("按 q 退出，按 s 保存当前画面")

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取画面失败")
        break
    
    # 推理
    label, confidence = predict(frame)
    
    # 显示结果
    cv2.putText(frame, f"Page: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 根据界面给出提示（简单版）
    if label == 'wechat_videochat':
        cv2.putText(frame, "Tip: Click camera icon", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    elif label == 'wechat_chat':
        cv2.putText(frame, "Tip: Type message here", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.imshow("WeChat Page Assistant", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"capture_{label}.jpg"
        cv2.imwrite(filename, frame)
        print(f"已保存: {filename}")

cap.release()
cv2.destroyAllWindows()
print("程序退出")
