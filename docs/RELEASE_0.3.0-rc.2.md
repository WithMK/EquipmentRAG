# EquipmentRAG 0.3.0-rc.2 Release Candidate

## 범위

이 Release Candidate는 `0.3.0-rc.1`의 Code/Document RAG, Local API/UI와
Retrieval 평가 기능에 다음 내용을 추가합니다.

- SQLite FTS5 기반 영속 BM25 Keyword Index
- ChromaDB `where_document.$contains` Exact Match Channel
- Semantic, Exact Match, FTS5 결과의 RRF 병합
- Code와 Document 증분 색인의 FTS5 추가·변경·삭제 동기화
- 설비 식별자(`E-024`, `ALM_204` 등) 검색 보정
- `/v1/answer`의 `conversation` 및 `retrieval_query` 계약
- FTS 준비 상태를 포함한 통합 CLI `status`

기본 `search.mode`는 실제 설비 평가 전까지 `semantic`을 유지합니다. FTS Index는
함께 구축되며 평가를 통과한 환경에서만 `hybrid`로 전환합니다.

## ContextManager 연결 경계

EquipmentRAG는 검색과 근거 기반 답변만 담당합니다.

- `question`: 현재 사용자의 원문 질문
- `retrieval_query`: ContextManager가 대명사와 생략어를 해소한 검색 질의
- `conversation`: 이전 `user`/`assistant` 대화, 최대 20개 Message

EquipmentRAG는 `session_id`, 대화 영속 저장, 장기 Memory, Session 만료와 삭제를
구현하지 않습니다. 이 기능은 다음 단계에서 ContextManager가 소유합니다.

## 연결 전 인수 확인

### 1. 설치와 설정

```powershell
python -m pip install --no-index --find-links=.\wheels -r requirements-offline.txt
Copy-Item .\config\config.offline.example.yaml .\config\config.local.yaml
python -m app status --config config\config.local.yaml
```

`Source`, `Embedding model`, `ChromaDB` 경로와 `Lexical` 상태를 확인합니다.

### 2. 전체 초기 색인

FTS5를 처음 활성화한 기존 환경은 설정 Fingerprint가 달라져 자동으로 전체 재색인
대상이 됩니다. 명시적으로 실행하려면 다음 명령을 사용합니다.

```powershell
python -m app index `
  --config config\config.local.yaml `
  --source-type all `
  --full
```

### 3. Retrieval 평가

사내 실제 File 이름과 질문은 GitHub가 아닌 폐쇄망 평가 Dataset에 저장합니다.

```powershell
python -m app evaluate `
  --config config\config.local.yaml `
  --dataset D:\EquipmentData\evaluation\retrieval.jsonl `
  --top-k 5 `
  --min-hit-rate 0.90 `
  --min-recall 0.80 `
  --min-mrr 0.70
```

`semantic` 결과를 기준선으로 저장한 뒤 `search.mode: hybrid`로 변경해 같은 Dataset을
다시 평가합니다. 세 지표가 유지되거나 개선되고 주요 Alarm/IO 질의의 오탐이 허용
범위일 때 Hybrid를 운영값으로 확정합니다.

### 4. Local API 계약 확인

```powershell
python -m app serve `
  --config config\config.local.yaml `
  --host 127.0.0.1 `
  --port 8765

Invoke-RestMethod http://127.0.0.1:8765/health
```

`POST /v1/retrieve`와 `POST /v1/answer` 요청 형식은
[`LOCAL_API.md`](LOCAL_API.md)를 따릅니다. 연결 전에는 Loopback에서 검증하고,
다른 PC에 노출할 때는 Firewall, TLS, 인증과 `--allow-remote` 정책을 별도로 적용합니다.

## 자동 검증 결과

- 전체 Unit/HTTP/Parser/ChromaDB/FTS5 Test 통과
- Python Source Compile 통과
- `git diff --check` 통과
- GitHub Actions CI 통과 필요

실제 폐쇄망 PC의 Model, 실제 설비 Index, LLM과 Retrieval 품질 검증은 배포 환경에서
완료해야 합니다.
