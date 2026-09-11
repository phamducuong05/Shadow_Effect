# Demo GPSDiffusion SDXL: ảnh + mask → FastAPI Docs

Demo này không cần DESOBAv2, train model hoặc `accelerate config`.
Giao diện Swagger cho upload hai file, gọi model và mở ảnh kết quả.

Mặc định: 512×512, FP16 cho UNet/ControlNet/geometry, VAE FP32,
1 ảnh/lần, 50 bước. Model được giữ trên RAM và chuyển lên GPU theo giai đoạn.
Hai text encoder được giải phóng sau khi mã hóa prompt cố định
`foreground object with shadow`. Các ứng viên chạy lần lượt, không ghép batch.
Hậu xử lý theo model gốc trả ảnh 256×256.

Thiết kế hướng tới server NVIDIA 16 GB, khoảng 15 GB VRAM trống. Đây không phải
cam kết mức VRAM: cần đo bằng checkpoint thật trên server. Chuẩn bị đủ RAM hệ
thống cho CPU offload (nên có ít nhất khoảng 32 GB) và dung lượng trống cho
checkpoint SDXL cùng cache Hugging Face (nên chừa khoảng 40 GB).

## 1. Upload code và weight

Nếu dùng gói `gpsdiffusion-sdxl-demo.zip` đi kèm, upload ZIP lên server rồi chạy:

```bash
mkdir -p /workspace
unzip gpsdiffusion-sdxl-demo.zip -d /workspace
cd /workspace/GPSDiffusion-Object-Shadow-Generation-SDXL
```

Thay `/workspace` bằng thư mục bạn có quyền ghi. Gói ZIP có code và hướng dẫn,
không chứa weight, cache model hoặc môi trường Conda.

Upload thư mục dự án hiện tại từ máy Windows lên server, ví dụ vào
`/workspace/GPSDiffusion-Object-Shadow-Generation-SDXL`.
Giữ các file `.py` và toàn bộ thư mục `ldm/`. Không cần upload `.git/`,
`.venv-demo-tests/`, `__pycache__/` hoặc ảnh so sánh trong README.
Nếu clone lại upstream thì phải chép thêm các file demo và hai file đã sửa
`base_network.py`, `attention_processor.py`; upstream chưa có demo này.

Đặt weight như sau, với mọi đường dẫn tính từ thư mục dự án trên server:

```text
GPSDiffusion-Object-Shadow-Generation-SDXL/
├── api.py
├── gps_sdxl_inference.py
├── check_demo.py
├── requirements-demo.txt
├── ldm/
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

`controlnet/` có thể chứa `.bin` hoặc nhiều shard thay vì một `.safetensors`;
giữ nguyên toàn bộ cấu hình, index và các shard trong bộ tải về. Không đổi tên
weight tùy ý và không tạo thêm lớp `pretrained_models/pretrained_models/`.

Có thể tạo thư mục trước khi upload trên Linux:

```bash
mkdir -p pretrained_models/controlnet models/pretrained_models
```

Lấy bộ **SDXL** ở link checkpoint trong [README gốc](README.md#installation):
[Baidu Cloud](https://pan.baidu.com/s/13NYGw3SS4B4n6mPtU1Me1Q?pwd=bcmi), mã `bcmi`.
Các tên/đường dẫn trên được xác định từ code; nội dung archive chưa được kiểm
tra trong môi trường phát triển này. `Shadow_cldm.ckpt` bản SD 1.5 không thay
thế cho `controlnet/` và `ip_adapter.ckpt` bản SDXL.

Nếu chỉ muốn ảnh thô, có thể bỏ `Shadow_ppp.ckpt`, đặt
`GPSXL_LOAD_POSTPROCESS=false` rồi chọn `apply_postprocess=false` trong Swagger.

Model nền `stabilityai/stable-diffusion-xl-base-1.0` sẽ được tải từ Hugging Face
lần đầu (UNet, VAE, hai text encoder/tokenizer, scheduler). Không cần SDXL refiner.
Các lần sau dùng cache. Muốn chạy offline, chuẩn bị đầy đủ model nền theo cấu
trúc Diffusers và cache dependency trước, rồi đặt `GPSXL_BASE_MODEL` thành
đường dẫn thư mục model nền đó.

## 2. Tạo Conda trên server Linux

Các lệnh dưới đây giả định server đã cài Conda và NVIDIA driver.
Thay đường dẫn `/workspace/...` bằng nơi bạn upload code:

```bash
cd /workspace/GPSDiffusion-Object-Shadow-Generation-SDXL
nvidia-smi
conda create -n gpsdiffusion-sdxl-demo python=3.10 -y
conda activate gpsdiffusion-sdxl-demo
python -m pip install --upgrade pip

# Bộ PyTorch/CUDA 12.1 theo phiên bản của repo.
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-demo.txt
python -m pip check
```

Không cài thêm `requirements.txt` của training, không clone Diffusers `main`
và không cài xFormers. Bộ demo dùng Diffusers 0.34.0 và PyTorch SDPA.
Bộ CUDA 12.1 này cần driver tương thích. Nếu GPU là thế hệ mới không được
PyTorch 2.4 hỗ trợ, cần chọn bộ PyTorch/torchvision tương thích GPU và điều
chỉnh hai dòng pin tương ứng trước khi cài; tránh cài CUDA wheel một cách ngẫu nhiên.

Nếu `conda activate` chưa được khởi tạo trong Bash:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpsdiffusion-sdxl-demo
```

Nếu server chạy Windows, mở **Anaconda Prompt**, dùng:

```bat
cd /d D:\Documents\GPSDiffusion-Object-Shadow-Generation-SDXL
conda create -n gpsdiffusion-sdxl-demo python=3.10 -y
conda activate gpsdiffusion-sdxl-demo
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-demo.txt
```

## 3. Kiểm tra trước khi chạy

Trên Linux, chọn GPU và nơi giữ cache (tùy chọn):

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/.cache/huggingface"
python check_demo.py
```

Sau `CUDA_VISIBLE_DEVICES=0`, thiết bị bên trong demo là `cuda:0`.
`check_demo.py` kiểm tra import, file weight, phép tính CUDA và VRAM còn trống.
Nó không tải model nền, không đọc nội dung checkpoint và không thay thế kiểm thử
inference thật. Chỉ tiếp tục sinh ảnh khi các mục `NOT READY` đã được xử lý.

## 4. Chạy FastAPI và mở /docs từ máy cá nhân

Chạy trên server, ở thư mục dự án:

```bash
conda activate gpsdiffusion-sdxl-demo
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

**Chỉ dùng một worker, không dùng `--reload`** để tránh nhiều bản model chiếm GPU.
Giữ terminal này chạy. Nhấn Ctrl+C để tắt.

Từ terminal trên máy cá nhân, mở SSH tunnel (thay `USER` và `SERVER_IP`):

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER_IP
```

Sau đó mở trình duyệt trên máy cá nhân:

- Swagger: <http://127.0.0.1:8000/docs>
- Trạng thái: <http://127.0.0.1:8000/health>

Nếu port 8000 trên máy cá nhân đang bận, dùng
`ssh -N -L 8001:127.0.0.1:8000 USER@SERVER_IP`, rồi mở port 8001.

Nếu server dùng dịch vụ notebook/cloud có port proxy, chạy với
`--host 0.0.0.0` và mở port 8000 bằng chức năng của nhà cung cấp.
Với mạng nội bộ cho phép truy cập trực tiếp cũng dùng `--host 0.0.0.0`,
rồi mở `http://SERVER_IP:8000/docs`. Demo không có xác thực; SSH tunnel là
cách mặc định để dùng riêng. Proxy đặt app dưới một tiền tố URL có thể cần
`--root-path` của Uvicorn để các link kết quả trỏ đúng.

## 5. Test ảnh và mask

1. Mở `POST /predict` → **Try it out**.
2. `image`: chọn ảnh ghép đã có vật thể cần tạo bóng.
3. `mask`: ảnh mask cùng kích thước; vật thể trắng, nền đen.
   Dùng PNG đen/trắng rõ ràng. Pixel mask ≥128 được coi là vật thể.
   Ảnh/mask có EXIF sẽ được chỉnh chiều trước khi kiểm tra kích thước.
4. Giữ `num_samples=1`, `num_steps=50`, `seed=42`, `apply_postprocess=true`.
5. Nhấn **Execute** và chờ. Request đầu nạp weight và có thể tải model nền nên
   chậm hơn các request sau. Nếu proxy có timeout ngắn, dùng SSH tunnel.

Response gồm:

- `generated`: link ảnh thô 512×512.
- `postprocessed`: link ảnh hậu xử lý 256×256, giữ nền ngoài vùng bóng dự đoán.
- `shadow_masks`: link mask bóng 256×256.
- `seeds`: seed từng ảnh; các ứng viên dùng `seed`, `seed+1`, ...
- `peak_vram_gb`: bộ nhớ CUDA reserved đỉnh của tiến trình trong lần generate,
  tính bằng GiB; không tính model load đầu tiên hoặc tiến trình GPU khác.

Mở các `url` trong response để xem/tải PNG. Kết quả lưu ở
`outputs/<request_id>/`. Thư mục này không tự xóa và cần được dọn khi không dùng.
Đầu vào được resize vuông như code gốc; ảnh không vuông có thể bị biến dạng.
Seed cố định giúp lặp lại trên cùng môi trường, không bảo đảm giống bit giữa GPU/version khác nhau.

Test qua curl trên server cũng được:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -F 'image=@/path/to/image.png' \
  -F 'mask=@/path/to/mask.png' \
  -F 'num_samples=1' -F 'num_steps=50' -F 'seed=42' \
  -F 'apply_postprocess=true'
```

## 6. Khi thiếu VRAM hoặc gặp lỗi

- **CUDA out of memory / HTTP 503**: xem `nvidia-smi`, dừng demo bằng Ctrl+C,
  khởi động lại với chế độ chuyển ControlNet và UNet lên GPU luân phiên
  ở từng bước. Chế độ này chậm hơn:

  ```bash
  export GPSXL_LOW_VRAM=true
  python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
  ```

  PowerShell dùng `$env:GPSXL_LOW_VRAM="true"`; Anaconda Prompt dùng
  `set GPSXL_LOW_VRAM=true`. Vẫn bắt đầu với một ảnh. Giảm số bước giúp giảm
  thời gian nhưng thường không giải quyết đỉnh VRAM của một bước.

- **`/health` → `not_ready`**: đọc `missing_files` và `cuda_available`.
  `ready_to_load` nghĩa là đủ file và thấy CUDA nhưng model chưa được nạp.
  `ok` nghĩa là model đã nạp; `busy` là đang xử lý; `error` chứa lỗi gần nhất.
  Kiểm tra file tồn tại không đồng nghĩa checkpoint đúng định dạng.
- **Sai weight / shape mismatch**: kiểm tra đúng bộ SDXL, ControlNet có
  `conditioning_channels=5`, IP-Adapter đi cùng checkpoint đó. Không bỏ qua
  lỗi load bằng `strict=False` vì có thể khiến model sinh ảnh với weight ngẫu nhiên.
- **Chưa có PPP**: đặt `GPSXL_LOAD_POSTPROCESS=false`, restart server,
  chọn `apply_postprocess=false`. Sau khi có PPP, bật lại và restart.
- **HTTP 409**: đang có request GPU khác. Đợi hoàn tất rồi thử lại.
- **HTTP 422**: sai ảnh/mask, mask rỗng hoặc tham số ngoài giới hạn.
- **HTTP 413**: ảnh hoặc mask quá 20 MiB. Demo giới hạn mỗi ảnh tối đa
  25 triệu pixel trước resize.
- **Lỗi Hugging Face**: kiểm tra mạng, dung lượng cache và quyền truy cập model.
  Request lỗi có thể thử lại; xem traceback tại terminal server.
- **Không thấy GPU sau khi cài**: kiểm tra đang ở đúng Conda và chạy
  `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.

Các biến cấu hình đọc khi server khởi động:

| Biến | Mặc định |
|---|---|
| `GPSXL_WEIGHTS_DIR` | `<repo>/pretrained_models` |
| `GPSXL_POST_CHECKPOINT` | `<repo>/models/pretrained_models/Shadow_ppp.ckpt` |
| `GPSXL_BASE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` hoặc thư mục Diffusers local |
| `GPSXL_DEVICE` | `cuda:0` |
| `GPSXL_LOAD_POSTPROCESS` | `true` |
| `GPSXL_LOW_VRAM` | `false` |
| `GPSXL_OUTPUT_DIR` | `<repo>/outputs` |

## 7. Kiểm chứng và giới hạn

```bash
python -m unittest discover -s tests -v
```

Tests CPU kiểm tra upload, trả/download PNG, xử lý lỗi, khóa request GPU,
mask/geometry và attention. API tests dùng model thay thế vì không có GPU/weight;
chúng không đánh giá chất lượng tạo bóng hoặc chứng minh vừa 15 GB VRAM.

Demo giữ cấu hình attention 4 IP tokens như script training/inference gốc.
Phần geometry sửa việc `fillPoly` tô vào bản copy và việc đổi width/height bằng
view tensor; centroid embedding được xây mới một lần thay vì cộng dồn theo
timestep. Những sửa này cần được đánh giá hình ảnh với weight thật; không cam
kết output giống hệt script dataset gốc. Script training/dataset không bị sửa.

Tham khảo: [PyTorch CUDA wheels](https://docs.pytorch.org/get-started/previous-versions/),
[Diffusers 0.34 memory](https://huggingface.co/docs/diffusers/v0.34.0/en/optimization/memory),
[FastAPI upload](https://fastapi.tiangolo.com/tutorial/request-files/).
