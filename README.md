# EquipmentRAG

Windows 폐쇄망에서 C# 설비 제어 소스코드를 색인하고 로컬 LLM으로 검색·분석하기 위한 최소 RAG PoC입니다.

## 현재 범위

Phase 4까지 프로젝트 구조, YAML 설정 로더, 로컬 임베딩, 영구 ChromaDB, C# 소스 탐색 및 구조 기반 Chunking을 제공합니다. 증분 색인과 LLM 연동은 이후 Phase에서 순차적으로 구현합니다.

## 요구 환경

- Windows 11
- Python 3.10 이상
- 인터넷 연결 없이 사용할 로컬 임베딩 모델 및 LLM 서버

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

폐쇄망 설치용 wheel은 외부망 PC에서 준비합니다.

```powershell
python -m pip download -r requirements.txt -d wheels
```

폐쇄망 PC에서는 다음과 같이 설치합니다.

```powershell
python -m pip install --no-index --find-links=.\wheels -r requirements-offline.txt
```

## 설정

[`config/config.yaml`](config/config.yaml)의 경로와 로컬 LLM 정보를 환경에 맞게 수정합니다. 상대 경로는 프로젝트 루트를 기준으로 해석됩니다.

설정 파일에는 API Key, 인증정보, 사내 경로 또는 실제 설비 자료를 넣지 마세요. 필요한 비밀값은 Git에 포함되지 않는 `.env` 등 별도 로컬 수단으로 관리합니다.

## Phase 1 검증

```powershell
python -m unittest discover -s tests -v
python -m app.config --config config\config.yaml
```

두 번째 명령은 비밀정보 없이 해석된 설정 요약을 JSON으로 출력합니다.

## Phase 2: 로컬 임베딩 모델

BGE-M3 등 Sentence Transformers 호환 모델을 `models/` 아래에 수동으로 배치하고 설정의 `embedding.model_path`를 해당 디렉터리로 지정합니다. `models/`는 Git에서 제외되며 실행 중 모델 다운로드를 시도하지 않습니다.

```yaml
embedding:
  model_path: ./models/bge-m3
  batch_size: 16
  device: null
  normalize_embeddings: true
```

`device: null`은 사용 가능한 장치를 자동 선택합니다. CPU로 고정하려면 `device: cpu`, CUDA를 지정하려면 예를 들어 `device: cuda:0`을 사용합니다.

실제 로컬 모델을 배치한 후 다음 명령으로 로딩과 벡터 생성을 확인합니다.

```powershell
python -m app.embedding.embedding_service `
  --config config\config.yaml `
  --task query `
  --text "Press Z축 Home 실패"
```

출력에는 모델 경로, 임베딩 차원, 벡터 길이와 일부 미리보기만 포함됩니다. 모델 파일이나 전체 벡터는 GitHub에 저장하지 않습니다.

## Phase 3: 영구 ChromaDB 벡터 저장소

`PersistentChromaStore`는 로컬 `PersistentClient`와 코사인 거리를 사용합니다. 임베딩은 애플리케이션에서 미리 계산해 전달하므로 ChromaDB가 외부 모델을 다운로드하지 않으며, 익명 텔레메트리도 비활성화합니다.

각 Chunk에는 다음 Metadata가 항상 저장됩니다.

```text
equipment, source_type, repository, project, file_name, file_path,
relative_path, class_name, method_name, chunk_index, file_hash,
modified_time, language
```

합성 C# 테스트 Chunk 두 개를 저장하고 LLM 없이 검색하려면 다음 명령을 실행합니다.

```powershell
python -m app.vectorstore.chroma_store `
  --config config\config.yaml `
  --seed-demo `
  --query "Z축 원점 복귀 실패"
```

같은 명령에서 `--seed-demo`를 빼고 다시 실행하면 기존 영구 컬렉션을 그대로 검색합니다. 생성되는 `data/chroma/`는 로컬 실행 데이터이며 Git에서 제외됩니다.

## Phase 4: C# 소스 탐색 및 Chunking

스캐너는 설정한 소스 폴더를 재귀 탐색하고 제외 디렉터리를 가지치기합니다. 확장자는 대소문자를 구분하지 않으며 UTF-8/BOM, UTF-16, CP949를 지원합니다. 각 파일의 원본 바이트 SHA-256과 UTC 수정 시각도 계산합니다.

청커는 Roslyn 없이 namespace, type, method, `#region`, XML/일반 주석을 가능한 범위에서 인식합니다. 메서드 경계를 우선 사용하고 긴 코드는 원본 줄 단위로 분할합니다. 구조를 찾지 못하면 안전한 크기 기반 Chunking으로 전환하며 모든 Chunk 내용은 원본 소스의 일부입니다.

```yaml
source:
  chunk_size: 4000
  chunk_overlap: 400
```

스캔 및 Chunk 메타데이터를 확인합니다. 기본 출력에는 소스 본문을 노출하지 않습니다.

```powershell
python -m app.chunkers.csharp_chunker --config config\config.yaml
```

합성 테스트 폴더 등 다른 경로를 일시적으로 확인하려면 `--source`를 사용합니다. 소스 본문 출력은 명시적으로 `--show-content`를 지정했을 때만 활성화됩니다.

## 데이터 보안

외부 GitHub 저장소에는 Framework와 합성 테스트 데이터만 저장합니다. 아래 항목은 `.gitignore`로 제외됩니다.

- 실제 C# 설비 소스: `data/source/`
- ChromaDB 데이터: `data/chroma/`
- 로컬 모델 및 GGUF: `models/`, `*.gguf`
- 오프라인 wheel: `wheels/`
- 로그와 비밀 설정: `logs/`, `*.log`, `.env*`
