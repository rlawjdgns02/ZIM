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

**원본 대비 변경 사항:**
- `.gitignore` 추가, README 교체
- `script/prepare_dataset.py` 추가: 외부 matting 데이터셋(AM-2K, PPM-100, AIM-500, Transparent-460)을 평가 형식(`data_list.txt` + 프롬프트 json)으로 변환
- `script/run_eval_dataset.sh` 추가: 변환된 데이터셋에서 ZIM vs SAM 평가
- `eval/main_eval.py`: 진행률 표시, 크기 그룹별 샘플 수(N) 출력, `--eval-save-dir` 지정 시 이미지별 점수 CSV와 요약 저장
- `script/run_eval.sh`: `data_root`를 `data`로 지정
- `reports/`: 실험 결과와 분석

## 실험 설계

### 핵심 질문

> **ZIM의 우위는 저자들이 만든 벤치마크(MicroMat-3K) 밖, 사람이 라벨링한 독립 데이터셋에서도 유지되는가?**

**이렇게 정한 이유:** MicroMat-3K의 정답은 저자들의 라벨 컨버터로 만든 pseudo-label을 사람이 수정한 것이고, ZIM의 학습 데이터(SA1B-Matte)도 같은 컨버터로 만들어졌습니다. 정답과 학습 데이터가 같은 파이프라인에서 나왔으므로 ZIM에 유리한 편향이 있을 수 있습니다. 학습 코드가 없어 재학습 ablation은 불가능하므로, **사전학습된 모델을 독립 데이터셋에서 zero-shot으로 평가**하는 것이 검증 가능한 범위입니다.

### 실험 구성

| 단계 | 검증할 주장 | 데이터셋 | 설계 이유 |
|---|---|---|---|
| **Phase 0** 재현 | 공식 체크포인트와 평가 코드가 논문 수치를 내는가 | MicroMat-3K (3,000장) | 평가 파이프라인이 논문과 같아야 새 데이터셋 결과를 논문 수치와 같은 기준으로 비교할 수 있음 |
| **E1** 일반화 | ZIM이 SAM보다 미세 구조(털, 머리카락)를 더 정밀하게 표현한다 | AM-2K validation (200장), PPM-100 (100장) | AM-2K는 동물 털이라 MicroMat-fine과 성격이 가장 비슷함. PPM-100은 **라벨 컨버터 학습에 쓰이지 않은** 데이터라 오염 없는 대조군 |
| **E7** 한계 (보조) | 논문이 밝힌 한계: 투명 객체에 약하다 | AIM-500 (500장, 카테고리 7개) | 같은 데이터셋 안에서 투명(34장)과 나머지를 비교해 라벨 스타일 차이를 통제. 유형 정보(SO/STM/NS)로 투명성과 비주류성을 분리 |
| E7 (주) | 같음 | Transparent-460 test | 논문 Table 7과 같은 데이터셋. 팀원 담당 |

### 공통 프로토콜과 이유

- **모델:** ZIM ViT-B (`zim_vit_b_2043`) vs SAM ViT-B (`sam_vit_b_01ec64`). 같은 백본 크기로 비교해야 개선이 ZIM의 기법에서 온 것인지 볼 수 있습니다.
- **프롬프트:** 새 데이터셋에는 프롬프트가 없으므로 [script/prepare_dataset.py](script/prepare_dataset.py)로 생성합니다. 규칙은 **MicroMat-3K가 제공한 프롬프트를 측정해서** 맞췄습니다. 그래야 논문 수치와 같은 조건으로 비교할 수 있습니다.
  - 박스: 전경 min–max에 좌표별 ±5% 균일 노이즈 (MicroMat 박스 300개 중 300개가 이 범위)
  - 점: positive/negative 개수를 MicroMat-3K 3,000개의 개수 분포에서 추출. positive는 α>0.5, negative는 α<0.02에서 선택
  - 이미지별 시드 고정. 모든 모델에 같은 프롬프트를 줍니다.
- **지표:** 논문과 같은 MSE, SAD, MAE, Grad, Conn과 크기 그룹(S/M/L). SAD는 해상도에 비례하므로 데이터셋 간 비교에는 MSE·MAE를 씁니다.
- **통계:** 평균은 소수의 큰 실패에 끌려가므로, 같은 이미지에 대한 쌍 비교(Wilcoxon), 이미지 단위 승률, 중앙값, 부트스트랩 95% CI를 함께 보고합니다. 실패는 MSE>100으로 정의했습니다.

## 핵심 결과

| 실험 | 결과 | 판정 |
|---|---|---|
| **Phase 0** | 논문과 비교 가능한 64개 수치가 모두 **±0.8% 이내**로 재현 | 파이프라인 신뢰 가능 |
| **E1** | 두 독립 데이터셋 모두에서 ZIM이 모든 지표·프롬프트에서 유의하게 우세 (이미지 승률 89% 이상, p<10⁻¹¹). 박스 MSE: AM-2K 10.76 vs 26.37, PPM-100 16.22 vs 58.14 | **지지됨 (조건부)** |
| **E7 보조** | 비투명 466장은 박스 승률 97%로 E1 재현. **투명 34장은 우위 소멸** (승률 59%, p=0.50) | **한계 확인** |

**Phase 0** ([보고서](reports/phase0_micromat3k_vitb.md))
- ZIM의 개선은 대부분 큰 객체(L)에서 나옵니다. fine·박스 기준 L 그룹은 10.7배, S 그룹은 1.6배입니다.
- 작은 coarse 객체에 박스를 줄 때는 SAM이 근소하게 낫습니다(MSE 0.180 vs 0.198). 논문 Table 6에도 같은 수치가 있지만 본문에서는 언급되지 않습니다.

**E1** ([보고서](reports/E1_report.html))
- **오염으로 설명되지 않음:** 라벨 컨버터 학습에 쓰이지 않은 PPM-100에서도 박스 기준 3.6배 우세합니다.
- **격차는 줄었음:** 큰 객체의 박스 MSE 개선이 MicroMat-3K fine의 10.7배에서 2.5~3.6배로 줄었습니다. MicroMat-3K가 ZIM에 유리하다는 가설과 부합하지만, 객체 종류와 해상도가 달라 단정할 수는 없습니다.
- **점 프롬프트의 실패는 점 개수가 좌우함:** PPM-100에서 ZIM 실패율은 점 1개일 때 78%, 7개 이상일 때 3%입니다.

**E7 보조** ([보고서](reports/E7_AIM500_report.html))
- 투명 객체에서는 두 모델 모두 박스 대부분을 전경으로 칠하거나 전경·배경을 뒤집는 **"박스 채움" 실패**를 보입니다(투명 56~59%, 비투명 0%). SAM에서 물려받은 공통 약점입니다.
- 비투명 비주류 객체에서는 ZIM이 여전히 92% 이기므로, 원인은 비주류성보다 투명성 쪽으로 보입니다(세부 그룹 12~21장, 잠정적).
- 논문은 "파인튜닝한 ZIM이 전문 모델보다 낮다"고만 했지만, zero-shot에서는 **SAM보다도 낫지 않습니다.**

**공통 한계**
- 평가 코드가 SAM 출력을 0/1 이진 마스크로 바꾸므로, 연속값을 내는 ZIM에 유리합니다. 특히 Grad 지표는 투명 객체에서 ZIM이 완전히 틀린 경우에도 100% 이겼으므로, Grad 우위의 일부는 출력 형식 효과일 수 있습니다. → **F1(SAM 연속값 출력 평가)로 분리 예정**
- 프롬프트 시드 1개, ViT-B만 평가했습니다.

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
