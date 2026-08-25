# EquipmentRAG Local API

Phase 18 API는 Context Orchestrator나 사내 도구가 EquipmentRAG 검색과 답변 기능을
JSON으로 호출할 수 있게 합니다. 외부 Web Framework를 추가하지 않으며 Python 표준
라이브러리만 사용합니다.

## 시작

기본값은 현재 PC에서만 접근 가능한 Loopback 주소입니다.

```powershell
python -m app serve `
  --config config\config.local.yaml `
  --host 127.0.0.1 `
  --port 8765
```

같은 주소의 `/` 경로에서는 폐쇄망 Local UI가 제공됩니다. UI 사용법은
`docs/LOCAL_UI.md`를 참고합니다.

`0.0.0.0`이나 다른 Network Interface에 바인딩하려면 `--allow-remote`를 명시해야
합니다. API 자체에는 사용자 인증과 TLS가 없으므로 신뢰할 수 없는 Network에 직접
노출하지 않습니다. 사내 Network 연동 시 Firewall, Reverse Proxy, TLS와 인증 정책을
별도로 적용합니다.

## Health Check

Health Check는 Embedding 모델이나 LLM을 로드하지 않습니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

## 근거 검색

`POST /v1/retrieve`는 LLM을 호출하지 않고 검색 근거만 반환합니다.

```powershell
$body = @{
  query = "Loader Vacuum 알람 조건"
  source_type = "all"
  top_k = 6
  include_content = $false
  filters = @{
    code = @{
      class_name = "LoaderController"
    }
    document = @{
      unit = "Loader"
      document_type = "Maintenance Manual"
    }
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8765/v1/retrieve `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

`source_type`은 `code`, `document`, `all` 중 하나입니다. 문서 Revision이 지정되지
않으면 기본적으로 `document_status=active`, `is_latest=true`가 적용됩니다.

지원 Filter:

- Code: `equipment`, `repository`, `relative_path`, `class_name`, `method_name`
- Document: `project`, `equipment`, `unit`, `document_type`, `revision`,
  `document_status`, `is_latest`, `document_id`, `file_extension`

원문 코드나 문서 본문은 기본 응답에서 제외됩니다. 신뢰할 수 있는 호출자가 실제
Context를 필요로 할 때만 `include_content=true`로 지정합니다.

## 근거 기반 답변

`POST /v1/answer`는 검색 후 설정된 로컬 llama.cpp 또는 Ollama를 호출합니다.

```json
{
  "question": "Loader Vacuum 알람 원인과 복구 절차는?",
  "source_type": "all",
  "top_k": 6,
  "temperature": 0.1,
  "max_tokens": 800,
  "include_content": false,
  "filters": {
    "document": {
      "unit": "Loader"
    }
  }
}
```

응답에는 `answer`, `model`, `finish_reason`, `usage`와 답변 생성에 제공된 `sources`가
포함됩니다. 검색 결과가 없으면 LLM을 호출하지 않고 `finish_reason=no_context`를
반환합니다.

## 안전 제한

- 요청 Body 최대 크기: 1 MiB
- JSON Object와 UTF-8만 허용
- 알 수 없는 Field 거부
- Model Service는 Source Type별로 Lazy Initialization 후 재사용
- 동시 Model 추론은 Process 안에서 직렬화
- 응답은 `Cache-Control: no-store`와 `X-Content-Type-Options: nosniff` 적용
- API를 통한 색인, 파일 수정, Command 실행 기능은 제공하지 않음

설비 Source와 문서 내용이 응답에 포함될 수 있으므로 API Log, Reverse Proxy Log와
호출 애플리케이션의 저장 정책도 사내 보안 기준에 맞춰야 합니다.
