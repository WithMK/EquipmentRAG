# EquipmentRAG Local UI

Phase 20 Local UI는 폐쇄망 PC의 Browser에서 Code와 Document를 검색하고 근거 기반
답변을 확인하는 화면입니다. Local API와 같은 Process에서 제공되며 별도 Web Server나
Node.js Runtime이 필요하지 않습니다.

## 시작

```powershell
python -m app serve `
  --config config\config.local.yaml `
  --host 127.0.0.1 `
  --port 8765
```

Browser에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8765/
```

화면을 여는 것과 Health Check만으로는 Embedding 모델이나 LLM이 로드되지 않습니다.
첫 검색 시 필요한 Retrieval Service가 초기화됩니다.

## 제공 기능

- 코드와 문서를 함께 검색하거나 한 종류만 선택
- LLM을 사용하는 답변 생성과 LLM 없는 근거 검색 전환
- Top-K 지정
- Unit, 문서 종류, Revision Filter
- Class, Method Filter
- 근거의 File, Score, Line, Section, Page, Slide, Sheet와 Cell 범위 표시
- 필요할 때만 Source 원문 표시
- 답변 Clipboard 복사
- 좁은 화면을 위한 반응형 Layout과 Keyboard Label

답변과 Source 내용은 HTML로 해석하지 않고 일반 Text로 표시하므로 문서나 Source에
포함된 Script가 실행되지 않습니다.

## 보안 특성

- 모든 CSS와 JavaScript는 Repository 내부 정적 파일 사용
- 외부 CDN, Font, Analytics 및 원격 이미지 요청 없음
- Content Security Policy에서 Script, Style과 Network 연결을 현재 Origin으로 제한
- `Cache-Control: no-store`, `nosniff`, Frame 차단 Header 적용
- 화면에서 색인, File 수정, Upload 또는 Command 실행 불가
- 기본 Loopback 바인딩으로 같은 PC에서만 접근

`--allow-remote`는 UI 인증을 추가하는 옵션이 아닙니다. API와 UI 자체에는 로그인과
TLS가 없으므로 다른 PC에 직접 노출하지 않습니다. 원격 사용이 필요하면 사내 Reverse
Proxy에서 사용자 인증, TLS, 접근 범위와 Log 정책을 적용해야 합니다.

## 운영 확인

화면 우측 상단 상태가 `로컬 연결 정상`인지 확인합니다. 오류가 표시되면 다음 순서로
점검합니다.

1. `python -m app status --config config\config.local.yaml`
2. `Invoke-RestMethod http://127.0.0.1:8765/health`
3. Browser 주소와 실행 Port 일치 여부
4. 검색 시에만 오류가 발생하면 Embedding 모델과 ChromaDB Index 상태
5. 답변 생성에서만 오류가 발생하면 llama.cpp 또는 Ollama 상태

실제 설비 화면 캡처나 질문 결과에는 내부 정보가 포함될 수 있으므로 외부 전송 전에
사내 보안 기준을 적용합니다.
