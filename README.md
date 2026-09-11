# Shadow Effect — GPSDiffusion SDXL FastAPI Demo

Demo tạo bóng cho vật thể bằng GPSDiffusion SDXL. Đầu vào gồm ảnh composite
và mask của vật thể; đầu ra gồm ảnh sinh thô, ảnh hậu xử lý và mask bóng.
Giao diện thử model được cung cấp qua FastAPI Swagger.

Repo này dành cho chạy inference demo trên GPU server. Không cần tải dataset,
không cần train model và không cần chạy `accelerate config`.

## Luồng xử lý

```text
Ảnh composite + object mask
        ↓
Geometry predictor + mask embeddings
        ↓
GPSDiffusion SDXL (ControlNet + IP-Adapter)
        ↓
Ảnh sinh 512×512
        ↓
Post-processing tùy chọn
        ↓
Ảnh hoàn thiện + shadow mask 256×256
```

## Yêu cầu khuyến nghị

- Linux GPU server với NVIDIA GPU 16 GB VRAM; nên còn khoảng 15 GB trống.
- Khoảng 32 GB RAM hệ thống cho CPU offload.
- Khoảng 40 GB ổ đĩa trống cho checkpoint và cache SDXL.
- Conda và NVIDIA driver tương thích CUDA.

## Clone repository

```bash
git clone https://github.com/phamducuong05/Shadow_Effect.git
cd Shadow_Effect
```

## Đặt model weights

```text
Shadow_Effect/
├── pretrained_models/
│   ├── controlnet/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   ├── ip_adapter.ckpt
│   ├── Shadow_cls.pth
│   ├── Shadow_reg.pth
│   └── Shadow_cls_label.pkl
└── models/
    └── pretrained_models/
        └── Shadow_ppp.ckpt
```

Thư mục `controlnet/` có thể chứa file `.bin`, file index và nhiều shard.
Hãy giữ nguyên toàn bộ cấu trúc từ archive checkpoint SDXL. Weight không được
đưa lên Git vì đã có trong `.gitignore`.

Checkpoint SDXL gốc: [Baidu Cloud](https://pan.baidu.com/s/13NYGw3SS4B4n6mPtU1Me1Q?pwd=bcmi),
mã truy cập `bcmi`.

Model nền `stabilityai/stable-diffusion-xl-base-1.0` được tải từ Hugging Face
ở lần chạy đầu và được lưu trong cache.

## Tạo môi trường Conda

```bash
conda create -n gpsdiffusion-sdxl-demo python=3.10 -y
conda activate gpsdiffusion-sdxl-demo
python -m pip install --upgrade pip

python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements-demo.txt
python -m pip check
python check_demo.py
```

`check_demo.py` kiểm tra dependency, CUDA, VRAM và danh sách weight còn thiếu
mà không nạp toàn bộ model.

## Chạy FastAPI

```bash
export CUDA_VISIBLE_DEVICES=0
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Chỉ chạy một Uvicorn worker vì mỗi worker sẽ tạo một bản model riêng trên GPU.

Nếu server ở xa, tạo SSH tunnel từ máy cá nhân:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER_IP
```

Mở:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

Trong `POST /predict`, upload:

- `image`: ảnh composite đã có vật thể.
- `mask`: cùng kích thước với ảnh; vật thể trắng, nền đen.
- `num_samples`: bắt đầu với `1`.
- `num_steps`: mặc định `50`.
- `seed`: mặc định `42`.
- `apply_postprocess`: `true` để sinh thêm ảnh hậu xử lý và mask bóng.

Kết quả được lưu tại `outputs/<request_id>/`. API trả URL cho từng ảnh và
`peak_vram_gb` để theo dõi mức VRAM đỉnh trong inference.

## Chế độ VRAM thấp

Nếu gặp CUDA out-of-memory, dừng server và chạy lại:

```bash
export GPSXL_LOW_VRAM=true
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Chế độ này chuyển ControlNet và UNet lên GPU luân phiên nên chậm hơn, nhưng
giảm lượng model cùng nằm trên VRAM. Vẫn nên để `num_samples=1` khi thử đầu tiên.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Các test CPU kiểm tra tiền xử lý ảnh/mask, geometry, attention, seed, offload,
API upload/download, xử lý lỗi và model hậu xử lý. Chất lượng ảnh và mức VRAM
thực tế cần được xác nhận bằng checkpoint thật trên GPU server.

Xem [DEMO.md](DEMO.md) để có hướng dẫn đầy đủ, xử lý lỗi thường gặp và toàn bộ
biến môi trường cấu hình.

## Nguồn và giấy phép

Dự án phát triển từ
[bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL)
và công trình:

> Haonan Zhao, Qingyang Liu, Xinhao Tao, Li Niu, Guangtao Zhai,
> “Shadow Generation Using Diffusion Model with Geometry Prior,” CVPR 2025.

Mã nguồn tuân theo giấy phép trong [LICENSE](LICENSE).
