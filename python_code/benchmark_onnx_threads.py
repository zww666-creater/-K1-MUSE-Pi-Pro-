import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/home/hnu/elder_ai/models/wechat_page.onnx"

def test_threads(num_threads):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    x = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # warm up
    for _ in range(3):
        session.run([output_name], {input_name: x})

    times = []

    for _ in range(10):
        t0 = time.time()
        session.run([output_name], {input_name: x})
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    print(f"threads={num_threads}, avg={avg:.2f} ms")


if __name__ == "__main__":
    print("Available providers:", ort.get_available_providers())

    for t in [1, 2, 4, 8]:
        test_threads(t)
