import gradio as gr
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =====================================================================
# 🛠️ ĐOẠN VÁ LỖI CỦA KERAS 3 (MONKEY PATCH)
original_init = keras.layers.Layer.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
keras.layers.Layer.__init__ = patched_init
# =====================================================================

# Tải model đã train
model = load_model("face_model.h5")

# Danh sách nhãn lớp nhận diện
class_labels = {
    0: "HoangKyAnh", 1: "Lê Quang Dũng", 2: "Lê Tuấn Thành", 3: "Lương Ngọc Thuận",
    4: "Ngô Quốc Trung", 5: "Nguyen Ngoc Bao", 6: "Nguyễn Hoàng Quế Châu",
    7: "Nguyễn Phạm Hoàng An", 8: "Nguyễn Thị Khánh Lê", 9: "Nguyễn Thị Ngọc Tuyết",
    10: "Nguyễn Tiến Mạnh", 11: "Nguyễn Việt Đức", 12: "Nguyễn Đặng Vinh Phúc",
    13: "Phạm Gia Thành Duy", 14: "Phạm Hứa Nhật Minh", 15: "Phạm Nguyễn Bảo Châu",
    16: "Phạm Phú Hoà", 17: "Trần Hải Yến", 18: "Vũ Quang Thái", 19: "Đinh Hữu Khánh Anh",
    20: "Đoàn Hùng", 21: "Đỗ An Phúc"
}

def predict_face(image):
    if image is None:
        return "Vui lòng tải lên một hình ảnh!"
    
    target_size = (128, 128) 
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    
    name = class_labels.get(predicted_class, "Không rõ danh tính")
    return f"Kết quả nhận diện: {name} (Độ chính xác: {confidence:.1f}%)"

demo = gr.Interface(
    fn=predict_face,
    inputs=gr.Image(type="pil", label="Chụp hoặc tải ảnh khuôn mặt lên"),
    outputs=gr.Textbox(label="Danh tính dự đoán"),
    title="Hệ thống Nhận diện Khuôn mặt lớp LTP001",
    description="Tải một bức ảnh cận mặt lên để mô hình nhận diện AI phân tích."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)