# Retrieval 품질 평가

Phase 19 평가 기능은 정답으로 기대하는 Source가 검색 결과 상위 K개 안에 들어오는지
측정합니다. LLM 답변 문장을 평가하는 기능이 아니므로 LLM Server를 실행하지 않아도
됩니다. Embedding 모델과 이미 생성된 ChromaDB Index는 필요합니다.

## Dataset 형식

Dataset은 한 줄에 하나의 JSON Object를 기록하는 UTF-8 JSONL 파일입니다.

```json
{"id":"loader-vacuum","query":"Loader Vacuum 알람 복구 절차","source_type":"document","top_k":5,"filters":{"document":{"unit":"Loader"}},"expected":[{"file_name":"LoaderManual.pdf","section":"Vacuum Alarm Recovery","revision":"Rev.3"}]}
```

필수 Field:

- `id`: Case를 식별하는 고유 이름
- `query`: 실제 사용자가 입력할 질문
- `expected`: 하나 이상의 기대 Source Matcher

선택 Field:

- `source_type`: `code`, `document`, `all`. 기본값은 `all`
- `top_k`: Case별 검색 개수
- `filters`: Local API와 동일한 Code/Document Metadata Filter

기대 Source Matcher는 지정한 모든 Field가 일치할 때 정답으로 판정합니다. 문자열은
대소문자를 구분하지 않는 정확 일치이며 `relative_path`는 `\`와 `/`를 동일하게
취급합니다.

지원 Matcher:

```text
record_id, source_type, file_name, relative_path,
class_name, method_name,
document_type, revision, document_status,
section, subsection, page, slide, sheet, cell_range
```

처음에는 File과 Method 또는 File과 Section처럼 안정적인 Metadata 두세 개만
사용하는 것이 좋습니다. `record_id`는 Chunking 설정 변경으로 달라질 수 있으므로
장기 회귀 Dataset에서는 신중하게 사용합니다.

Repository의 `examples/evaluation.sample.jsonl`은 형식 예시이며 실제 File 이름과
Section을 폐쇄망 자료에 맞게 변경해야 합니다. 실제 설비 정보가 들어간 평가 Dataset은
GitHub에 Push하지 않습니다.

## 실행

```powershell
python -m app evaluate `
  --config config\config.local.yaml `
  --dataset D:\EquipmentData\evaluation\retrieval.jsonl `
  --top-k 5
```

JSON 결과도 별도 보관할 수 있습니다.

```powershell
python -m app evaluate `
  --config config\config.local.yaml `
  --dataset D:\EquipmentData\evaluation\retrieval.jsonl `
  --json `
  --output D:\EquipmentData\evaluation\results\baseline.json
```

## Metric

- `Hit@K`: Case별로 기대 Source가 하나라도 검색된 비율
- `Recall@K`: 전체 기대 Source 중 상위 K개에서 찾은 비율
- `MRR`: 각 Case에서 첫 정답 Source 순위의 역수 평균

Case마다 `top_k`가 다를 수 있으므로 결과에는 실제 사용한 K가 함께 기록됩니다.
세 Metric 모두 0에서 1 사이이며 높을수록 좋습니다.

## 품질 기준 적용

최소 품질 기준을 지정하면 하나라도 미달할 때 Process Exit Code가 `1`이 됩니다.

```powershell
python -m app evaluate `
  --config config\config.local.yaml `
  --dataset D:\EquipmentData\evaluation\retrieval.jsonl `
  --min-hit-rate 0.90 `
  --min-recall 0.80 `
  --min-mrr 0.70
```

모델이나 Chunking 설정을 변경하기 전에 Baseline 결과를 저장하고, 변경 후 같은
Dataset으로 다시 측정합니다. 실제 설비 자료가 없는 외부 GitHub CI에서는 단위 테스트만
실행하고 실제 품질 평가는 폐쇄망에서 수행합니다.

## Dataset 작성 원칙

- 실제 작업자가 자주 묻는 질문을 사용
- Alarm, IO, Sequence, Interlock, 복구 절차를 균형 있게 포함
- 코드 전용, 문서 전용, 통합 질문을 분리
- 최신 문서와 명시적 과거 Revision 질문을 모두 포함
- 검색 결과를 사람이 검토한 후 기대 Source로 등록
- 같은 의미의 한국어·영어·약어 질문을 포함
- Dataset 변경 이력과 평가에 사용한 Index/Model Version을 별도로 기록
