import time
import cv2
import numpy as np
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

last_time = time.time()
frame_count = 0
fps_window = []
video_writer = None

VIDEO_PATH = "segmentation_overlay.mp4"
VIDEO_SIZE = (640, 480)

def segmentation_callback(request: CompletedRequest):
    global last_time, frame_count, fps_window, video_writer

    try:
        start = time.time()

        metadata = request.get_metadata()
        outputs = imx500.get_outputs(metadata)
        if outputs is None:
            print("No outputs")
            return

        output = outputs[0]
        mask = (output[0] > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, VIDEO_SIZE, interpolation=cv2.INTER_NEAREST)

        frame = request.make_array("main")

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = cv2.resize(frame, VIDEO_SIZE)
        overlay = frame.copy()

        alpha = 0.5
        blue_channel = overlay[:, :, 0].astype(np.float32)
        mask_float = mask_resized.astype(np.float32) * alpha
        blue_channel = np.clip(blue_channel + mask_float, 0, 255)
        overlay[:, :, 0] = blue_channel.astype(np.uint8)

        cv2.circle(overlay, (VIDEO_SIZE[0] // 2, VIDEO_SIZE[1] // 2), 5, (0, 0, 255), -1)

        cv2.imshow("Segmentation Overlay", overlay)
        cv2.waitKey(1)
        video_writer.write(overlay)

        end = time.time()
        latency_ms = (end - start) * 1000
        frame_count += 1
        fps_window.append(end)
        fps_window = [t for t in fps_window if end - t <= 1.0]
        fps = len(fps_window)

        print(f"Latency: {latency_ms:.2f} ms | FPS: {fps}")

    except Exception as e:
        print("Error in callback:", e)


if __name__ == "__main__":
    imx500 = IMX500("/usr/share/imx500-models/picosam2_student_quantized.rpk")
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "segmentation"
    intrinsics.update_with_defaults()

    VIDEO_FPS = intrinsics.inference_rate
    print(f"Recording at model-defined FPS: {VIDEO_FPS}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, VIDEO_SIZE)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": VIDEO_FPS},
        buffer_count=12
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    picam2.pre_callback = segmentation_callback

    print("Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        picam2.stop()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()