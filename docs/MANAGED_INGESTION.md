# 자료 등록 · 자동 인덱싱 (0.3.0-rc.3)

A/O 프로젝트 화면에서 받은 파일을 이 서버가 보관하고 자동 색인합니다.
파일 업로드 → SQLite 영속 대기열 → 기존 파서·Chunker·로컬 Embedding →
Chroma 문서 Collection 및 설정된 SQLite FTS5 인덱스 순서입니다.
C/M과 llama.cpp를 호출하거나 모델을 학습하지 않습니다.

## 실행

기존 `document.enabled: true`, 로컬 Embedding 모델, 문서 Collection 설정을 사용합니다.
독립 키워드 검색까지 갱신하려면 기존 `search.lexical_backend: sqlite_fts5`와
`search.lexical_path`를 설정하고 검색 모드를 `hybrid`로 사용하세요.
다른 lexical 설정도 지원하지만 별도 FTS5 인덱스는 갱신하지 않습니다.

```powershell
# 두 서비스에 동일한 24자 이상의 비밀 값을 환경변수로 설정합니다.
$env:EQUIPMENT_RAG_INGESTION_KEY = '<운영 환경의 24자 이상 비밀 값>'
python -m app.api --config config/config.yaml --ingestion-root D:\EquipmentRAG-data\uploads
```

`--ingestion-root`가 없으면 쓰기 API는 비활성입니다. 키는 검색 API 키와 별개이고
브라우저에 전달하지 않습니다. 다른 PC에서 연결할 때는 기존 `--host`와
`--allow-remote` 설정이 필요합니다. 이 서비스 자체에는 TLS 종단 기능이 없으므로
루프백 외 배포의 암호화·접근 범위는 운영 환경에서 구성합니다.

자료 디렉터리는 CLI의 `document.source_paths` 외부에 두세요. 서버 한 프로세스만
같은 업로드 디렉터리를 사용해야 하며, CLI 인덱싱 작업은 API를 중지하고 실행하세요.
업로드 인덱스와 journal의 위치·모델·문서 처리 설정이 바뀌면 시작을 거부합니다.
이는 다른 인덱스를 `ready`로 잘못 표시하는 것을 막습니다. 이번 버전은 설정 변경에
따른 일괄 마이그레이션 UI를 제공하지 않습니다.

## API 계약

모든 아래 요청에 `Authorization: Bearer <EQUIPMENT_RAG_INGESTION_KEY>`가 필요합니다.
인증은 HTTP 요청 본문을 읽기 전에 확인합니다.

| API | 용도 |
| --- | --- |
| `GET /v1/ingestion/{namespace}/documents` | 해당 namespace 자료·상태 목록 |
| `POST /v1/ingestion/{namespace}/documents` | 업로드·동일 파일 확인·변경 파일 교체; 202 |
| `POST /v1/ingestion/{namespace}/documents/{id}/retry` | 실패한 현재 버전 재시도; 202 |

업로드 본문은 JSON이며 `filename`, `content_base64`, `project`, `equipment`, `version`을
사용합니다. `equipment`가 비어 있으면 RAG 설정의 기본 값을 사용합니다.
새 파일은 `version: 0`, 교체는 목록에서 확인한 현재 버전을 보냅니다.
재시도 본문은 `{"version": 1}`입니다. namespace는 영문·숫자·`_`·`-`, 1~100자입니다.
경로·서버 URL·실행 명령을 파일 이름으로 전달할 수 없습니다.

- 지원: MD, TXT, PDF, DOCX, PPTX, XLSX. 파일당 1 byte~10 MiB, JSON 최대 14 MiB.
- Office ZIP은 펼친 크기 100 MiB, 항목 10,000개까지 허용합니다.
- PDF 스캔본·이미지는 기존 OCR 설정과 오프라인 도구가 있어야 처리됩니다.
  추출할 텍스트가 없으면 실패로 표시합니다.
- namespace + 대소문자 구분 없는 파일 이름이 자료의 ID입니다. 이름이 다르면 별도 자료입니다.
- 같은 SHA-256은 중복 인덱싱하지 않습니다. 실패 상태의 같은 파일은 재시도 API를 사용합니다.
- 다른 내용은 현재 버전이 일치할 때만 교체하며, 처리 중 교체는 409입니다.
- `queued`, `indexing`, `ready`, `failed`를 반환합니다. `ready`는 인덱스 저장 완료이며
  모든 질문의 검색 성공이나 답변 정확성을 의미하지 않습니다.

## 복구와 데이터 보존

원본 최신 파일과 작업 상태는 `ingestion.db`에 저장합니다. 인덱싱 중 사용할 임시 파일은
작업 종료 시 정리합니다. 정상 종료는 실행 중 작업을 마칠 때까지 기다리며,
프로세스 강제 종료 후 시작하면 `indexing`을 대기열로 복구합니다.
실패 작업은 명시적 재시도 전까지 자동 반복하지 않습니다. Windows OS 파일 잠금은
프로세스 종료 시 해제되고, SQLite 연결은 명시적으로 닫습니다.

Chroma와 FTS5 사이에는 공통 트랜잭션이 없습니다. 변경할 Chunk ID를 먼저 기록하고,
재시도 시 두 인덱스가 같은 최신 자료로 수렴하도록 갱신·정리합니다. 중간 쓰기 실패가
있으면 API는 영향받는 project/equipment 문서 검색과 답변에 503을 반환합니다.
다른 프로젝트나 코드 전용 검색은 계속 가능합니다. 정상 재시도로 차단을 해제합니다.
파싱·Embedding 단계에서 실패하면 기존 인덱스는 유지되며 이전 내용이 검색될 수 있습니다.
목록의 `indexed_sha256`으로 마지막 성공 내용을 확인할 수 있습니다.
이 일관성 보호는 이 API 인스턴스에 적용됩니다. 별도 조회 프로세스/CLI는 journal을
참조하지 않으므로 복구 전 같은 인덱스를 별도로 조회하지 마세요.

CLI 문서 인덱서는 최초 상태가 없다고 equipment 전체를 지우지 않도록 변경했습니다.
기존 state에 기록된 Chunk만 삭제하므로 업로드 자료를 보호합니다. 반대로 state를
잃으면 과거의 미추적 Chunk를 자동 청소하지 않습니다. 인덱스와 상태 파일을 함께
백업·복구하세요. 코드 전체 재색인도 FTS의 `source_type=code`만 초기화합니다.

이번 범위에는 폴더 감시, 자료 삭제 UI, 과거 원본 버전 보관, 사용자별 권한,
자동 문서 생성·데이터셋 구축 및 품질 평가가 포함되지 않습니다.

## 검증

`python -m unittest discover -s tests -v`

`test_ingestion.py`는 실 SQLite FTS5와 HTTP 서버로 중복 등록, 교체, 프로젝트 분리,
부분 실패 복구, 중단 복구, Windows 파일 해제, 기존 CLI 자료 보존을 검증합니다.
Embedding은 합성 fixture입니다. 실제 폐쇄망 모델 및 파일별 추출 품질은 대상 PC에서
별도로 확인해야 합니다. 서버 로그에는 운영자 진단용 상세 오류가 기록됩니다.
