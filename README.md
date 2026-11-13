# ZIM 재평가: 다른 데이터셋에서의 Zero-Shot Image Matting 검증

> **출처 (Original Work)**
> 이 저장소는 NAVER Cloud의 [ZIM](https://github.com/naver-ai/ZIM) 공식 코드를 기반으로 수정한 것입니다.
>
> - 논문: Beomyoung Kim, Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, Joonsang Yu.
>   **"ZIM: Zero-Shot Image Matting for Anything"**, ICCV 2025. [arXiv:2411.00626](https://arxiv.org/abs/2411.00626)
> - 원본 코드: https://github.com/naver-ai/ZIM (Copyright (c) 2024-present NAVER Cloud Corp.)
> - 라이선스: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — 비영리 목적으로만 사용 가능합니다. 자세한 내용은 [LICENSE](LICENSE), [NOTICE](NOTICE)를 참고하세요.
> - 일부 코드는 [Segment Anything](https://github.com/facebookresearch/segment-anything) (Apache 2.0)을 기반으로 합니다.
>
> 원본 README와 사용법은 [원본 저장소](https://github.com/naver-ai/ZIM#readme)에서 확인할 수 있습니다.

## 용도

강의 실습 프로젝트입니다. 논문이 MicroMat-3K에서 보인 주장이 **다른 데이터셋에서도 성립하는지** 사전학습된 ZIM으로 zero-shot 평가하여 검증합니다.

- 공개된 코드는 추론/평가용(ONNX 가중치)이며 학습 코드가 없으므로, 재학습이 필요한 실험(라벨 컨버터, SA1B-Matte 등)은 다루지 않습니다.
- 모델 가중치와 데이터셋은 이 저장소에 포함되어 있지 않습니다. 아래 안내에 따라 원 출처에서 받아야 합니다.

**원본 대비 변경 사항:** `.gitignore` 추가, README 교체. (실험 코드가 추가되면 여기에 기록합니다.)

## 환경 설정

테스트 환경: Ubuntu 24.04, RTX 3090, NVIDIA 드라이버 595 (시스템 CUDA Toolkit 설치 불필요)

```bash
# 1) conda 환경
conda create -n zim python=3.10 -y
conda activate zim

# 2) PyTorch (CUDA 12.4 + cuDNN 9 포함)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# 3) ONNX Runtime GPU
pip install onnxruntime-gpu==1.20.2

# 4) 기타 의존성
pip install easydict opencv-python scikit-image pillow tqdm matplotlib huggingface_hub
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5) ZIM 설치
pip install -e .

# 6) ONNX Runtime이 torch에 포함된 cuDNN/CUDA 라이브러리를 찾도록 경로 등록 (최초 1회)
N=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
conda env config vars set LD_LIBRARY_PATH=$N/cudnn/lib:$N/cublas/lib:$N/cuda_runtime/lib:$N/cufft/lib:$N/curand/lib:$N/cuda_nvrtc/lib
conda deactivate && conda activate zim
```

### 가중치 및 데이터 다운로드

```bash
mkdir -p results data

# ZIM (Hugging Face) — encoder.onnx / decoder.onnx 가 든 폴더
hf download naver-iv/zim-anything-vitb --local-dir results/ --include "zim_vit_b_2043/*"
hf download naver-iv/zim-anything-vitl --local-dir results/ --include "zim_vit_l_2092/*"

# SAM (비교용, Meta 공식)
wget -P results https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# MicroMat-3K 테스트셋
wget -P data https://github.com/naver-ai/ZIM/releases/download/testset-v1/MicroMat3K.tar
tar -xf data/MicroMat3K.tar -C data
```

### 설치 확인

```bash
python -c "
from zim_anything import zim_model_registry
m = zim_model_registry['vit_b'](checkpoint='results/zim_vit_b_2043').cuda()
print(m.encoder.ort_session.get_providers())
"
# ['CUDAExecutionProvider', 'CPUExecutionProvider'] 가 출력되면 정상
```
