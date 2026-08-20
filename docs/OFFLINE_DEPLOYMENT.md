# EquipmentRAG 폐쇄망 반입·설치·운영 가이드

이 문서는 외부 개발용 GitHub Repository를 Windows 폐쇄망으로 반입하고, Python Package와 로컬 모델만으로 EquipmentRAG를 실행하는 절차를 설명합니다. 실제 설비 Source, 사내 문서, ChromaDB와 운영 로그는 외부망에 두지 않습니다.

## 1. 반입 원칙

외부망과 폐쇄망의 자산을 다음처럼 분리합니다.

```text
GitHub (Framework와 합성 테스트만)
  ↓ Git bundle 또는 승인된 Repository Archive
반입 매체
  ├─ repository/EquipmentRAG-main.bundle
  ├─ wheels/*.whl
  ├─ models/embedding/bge-m3/*
  ├─ models/llm/*.gguf
  ├─ installers/Python 및 LLM Runtime 설치 파일
  └─ SHA256SUMS.txt
  ↓ 보안 검사와 Hash 검증
폐쇄망
  ├─ 사내 Gitea: Framework Source
  ├─ D:/OfflineAssets: 모델과 설치 자산
  ├─ D:/EquipmentData: 실제 설비 Source와 문서
  └─ D:/OfflineRuntime: ChromaDB와 로그
```

Repository에 포함하면 안 되는 항목:

- 실제 설비 C# Source와 사내 문서
- `.env`, Token, API Key와 인증서
- Embedding 모델, GGUF와 Ollama Blob
- ChromaDB, 색인 상태, 로그와 운영 Backup
- Python wheel 및 설치 실행 파일

모든 모델은 사용 권한과 사내 반입 승인을 별도로 확인합니다.

## 2. 외부망 준비 PC 기준

가능하면 폐쇄망 대상 PC와 동일한 조건의 외부망 Windows PC에서 준비합니다.

- 동일한 Windows Architecture(일반적으로 x64)
- 동일한 Python Major/Minor Version
- 충분한 디스크 공간: wheel, BGE-M3, GGUF와 복사본 포함
- Git, Python과 Hash 계산이 가능한 PowerShell

환경 정보를 기록합니다.

```powershell
python --version
python -c "import platform; print(platform.platform()); print(platform.machine())"
git --version
```

## 3. Repository를 Git bundle로 준비

Commit되지 않은 파일은 bundle에 포함되지 않습니다. 먼저 검증된 `main`과 Clean 상태를 확인합니다.

```powershell
git switch main
git status --short
git log -1 --oneline
```

Git bundle은 Network Server 없이 Git Object와 Ref를 이동할 수 있습니다.

```powershell
New-Item -ItemType Directory -Force ..\offline-bundle\repository
git bundle create ..\offline-bundle\repository\EquipmentRAG-main.bundle main
git bundle verify ..\offline-bundle\repository\EquipmentRAG-main.bundle
git bundle list-heads ..\offline-bundle\repository\EquipmentRAG-main.bundle
```

필요한 Tag까지 전달하려면 검토 후 `main --tags`를 사용합니다. 개발 Branch 전체를 무조건 반입하지 않습니다.

## 4. Windows용 Python wheel 수집

빈 `wheels` 디렉터리에 현재 Windows/Python과 호환되는 Binary Package와 의존성을 수집합니다. Source Distribution은 폐쇄망에서 Compiler를 요구할 수 있으므로 우선 차단합니다.

```powershell
New-Item -ItemType Directory -Force ..\offline-bundle\wheels | Out-Null

python -m pip download `
  --only-binary=:all: `
  --dest ..\offline-bundle\wheels `
  --requirement requirements-offline.txt
```

특정 Package에 Windows wheel이 없어서 실패하면 해당 Package를 승인된 Build PC에서 별도로 wheel로 Build합니다. 실패한 상태에서 Source Archive만 반입하지 않습니다.

수집 결과에 Source Archive가 없는지 확인합니다.

```powershell
Get-ChildItem ..\offline-bundle\wheels -File |
  Where-Object { $_.Extension -ne ".whl" }
```

아무것도 출력되지 않아야 합니다. wheel 디렉터리는 매 Release마다 새로 만들고 이전 Version과 섞지 않습니다.

## 5. 모델과 Runtime 준비

### 5.1 Embedding 모델

BGE-M3 Sentence Transformers 디렉터리 전체를 다음 예시 위치에 복사합니다.

```text
offline-bundle/models/embedding/bge-m3/
```

최소한 모델 설정, Tokenizer, Module 설정과 Weight 파일이 함께 있어야 합니다. EquipmentRAG는 `local_files_only=True`를 사용하므로 누락 파일을 인터넷에서 자동 보충하지 않습니다.

### 5.2 llama.cpp

승인된 `llama-server.exe`, 필요한 Runtime DLL과 GGUF를 함께 준비합니다.

```text
offline-bundle/installers/llama.cpp/
offline-bundle/models/llm/model.gguf
```

### 5.3 Ollama

승인된 Windows Ollama 설치 파일과 GGUF를 준비합니다. Repository의 `deploy/ollama/Modelfile.example`을 GGUF 옆에 복사하고 `Modelfile`로 이름을 바꿉니다.

```text
FROM ./model.gguf
```

폐쇄망에서 다음 명령으로 로컬 모델을 생성합니다.

```powershell
Set-Location D:\OfflineAssets\models\llm
ollama create equipment-local -f .\Modelfile
ollama list
```

`-cloud` 이름의 모델은 폐쇄망 구성에 사용하지 않습니다.

## 6. 반입 자산 SHA-256 생성

모든 파일 준비가 끝난 뒤 Manifest 자체를 제외하고 Hash를 생성합니다.

```powershell
$bundleRoot = (Resolve-Path ..\offline-bundle).Path
$manifest = Join-Path $bundleRoot "SHA256SUMS.txt"

Get-ChildItem -LiteralPath $bundleRoot -File -Recurse |
  Where-Object { $_.FullName -ne $manifest } |
  Sort-Object FullName |
  ForEach-Object {
    $relative = $_.FullName.Substring($bundleRoot.Length + 1).Replace("\", "/")
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $relative"
  } | Set-Content -LiteralPath $manifest -Encoding utf8
```

Manifest와 반입 매체는 사내 보안 절차에 따라 Malware 검사와 승인을 받습니다.

## 7. 폐쇄망에서 Hash 검증

반입 후 설치 전에 모든 Hash를 다시 계산합니다.

```powershell
$bundleRoot = (Resolve-Path D:\Transfer\offline-bundle).Path
$manifest = Join-Path $bundleRoot "SHA256SUMS.txt"
$failed = @()

Get-Content -LiteralPath $manifest | ForEach-Object {
  $expected, $relative = $_ -split "  ", 2
  $target = Join-Path $bundleRoot $relative
  if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    $failed += "missing: $relative"
  } else {
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { $failed += "mismatch: $relative" }
  }
}

if ($failed.Count -gt 0) {
  $failed
  throw "Offline bundle integrity verification failed."
}
"All offline bundle hashes verified."
```

누락 또는 불일치가 하나라도 있으면 설치하지 않고 반입 원본을 다시 확인합니다.

## 8. Repository 복원과 사내 Gitea 이전

Git bundle에서 Working Copy를 만듭니다.

```powershell
Set-Location D:\Workspace
git clone --branch main `
  D:\Transfer\offline-bundle\repository\EquipmentRAG-main.bundle `
  EquipmentRAG
Set-Location .\EquipmentRAG
git status --short --branch
git log -1 --oneline
```

사내 Gitea에 빈 Repository를 만든 뒤 사내 Remote를 추가합니다. 외부 GitHub Credential을 반입하지 않습니다.

```powershell
git remote remove origin
git remote add intranet https://gitea.example.invalid/team/EquipmentRAG.git
git push -u intranet main
```

`gitea.example.invalid`은 문서용 Placeholder입니다. 실제 URL과 인증은 사내 관리 절차를 따릅니다.

## 9. 완전 오프라인 Python 설치

폐쇄망 PC에 승인된 Python을 설치한 뒤 프로젝트 전용 가상환경을 만듭니다.

```powershell
Set-Location D:\Workspace\EquipmentRAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install `
  --no-index `
  --find-links D:\Transfer\offline-bundle\wheels `
  --requirement requirements-offline.txt

python -m pip check
```

`--no-index`를 반드시 사용합니다. 설치 중 Package Index나 외부 URL 접근을 요구하면 반입 wheel이 불완전한 것입니다.

## 10. 로컬 운영 설정

추적되는 기본 설정을 직접 수정하는 대신 로컬 전용 파일을 만듭니다.

```powershell
Copy-Item .\config\config.offline.example.yaml .\config\config.local.yaml
notepad .\config\config.local.yaml
```

`config/config.local.yaml`은 `.gitignore`에 포함됩니다. 다음 값을 폐쇄망 PC에 맞게 변경합니다.

- `equipment.name`
- 실제 설비 Source가 있는 `source.path`
- BGE-M3 디렉터리인 `embedding.model_path`
- 운영 DB인 `chromadb.path`
- llama.cpp 또는 Ollama의 `llm` 설정
- 운영 Log 경로

Ollama를 사용할 때는 다음 부분으로 바꿉니다.

```yaml
llm:
  provider: ollama
  base_url: http://127.0.0.1:11434/api
  model: equipment-local:latest
  request_timeout_seconds: 120
```

설정 파일에 Token, Password 또는 실제 Source 내용을 기록하지 않습니다.

## 11. Offline 강제 실행과 단계별 검증

현재 PowerShell Session에서 Hugging Face Offline 환경변수를 설정합니다.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
```

먼저 Framework 회귀 테스트와 설정을 확인합니다.

```powershell
python -m unittest discover -s tests -v
python -m app.config --config .\config\config.local.yaml
python -m app.embedding.embedding_service `
  --config .\config\config.local.yaml `
  --task query `
  --text "Offline embedding 확인"
```

실제 Source를 변경하지 않는 Dry-run부터 실행합니다.

```powershell
python -m app.indexer --config .\config\config.local.yaml --dry-run
```

검토 후 색인, 검색과 RAG를 순서대로 실행합니다.

```powershell
python -m app.indexer --config .\config\config.local.yaml
python -m app.search "Z축 원점 복귀 실패" `
  --config .\config\config.local.yaml
python -m app.rag_service "Z축 원점 복귀 실패 원인을 설명해줘." `
  --config .\config\config.local.yaml `
  --top-k 5 `
  --max-tokens 512
```

합격 기준:

- 외부 Network 없이 모든 Python Import와 테스트 성공
- Embedding Vector 길이가 로컬 모델 차원과 일치
- Dry-run에서 대상 파일과 Chunk 수 확인
- 색인 재실행 시 변경 없는 파일 Skip
- Semantic Search에 파일·클래스·메서드·라인 표시
- RAG 답변에 `[S1]` 형식의 출처 표시
- 모델, Source, ChromaDB와 Log가 Git 상태에 나타나지 않음

## 12. 운영 Backup과 Update

Backup 대상:

- 사내 Gitea Repository
- `config/config.local.yaml`의 승인된 별도 Backup
- ChromaDB 디렉터리와 `index-state-*.json`
- Embedding/LLM 모델 원본과 SHA-256 Manifest
- Python wheel Bundle과 설치 Version 기록

Framework Update는 새 Git bundle과 새 wheel 디렉터리로 반입합니다. 기존 운영 디렉터리를 덮어쓰기 전에 새 가상환경에서 테스트합니다. Chunk, Embedding 모델 또는 Metadata Schema가 변경되면 인덱서가 전체 재색인을 수행할 수 있으므로 ChromaDB Backup과 디스크 여유 공간을 먼저 확인합니다.

Rollback은 이전 Git Commit, 이전 wheel Bundle, 이전 모델과 ChromaDB Backup을 하나의 Release 단위로 복원합니다. 서로 다른 Release의 DB와 Embedding 모델을 혼합하지 않습니다.

## 13. 문제 해결

### `No matching distribution found`

대상 Windows/Python과 호환되는 wheel이 빠졌습니다. 동일 환경의 외부망 PC에서 빈 wheel 디렉터리로 다시 수집합니다.

### 로컬 Embedding 모델을 찾지 못함

`embedding.model_path`가 실제 모델 디렉터리를 가리키는지와 모델 파일 Hash를 확인합니다. 인터넷 다운로드로 우회하지 않습니다.

### ChromaDB Dimension 불일치

현재 DB를 만든 Embedding 모델과 설정 모델이 다릅니다. Backup 후 올바른 모델로 전체 재색인합니다.

### llama.cpp 또는 Ollama 연결 실패

Localhost Port, Process, Model 이름과 `llm.base_url`을 확인합니다. 방화벽에서 외부 연결을 허용하는 방식으로 해결하지 않습니다.

### RAG 답변에 근거가 없음

먼저 `app.search`로 검색 품질과 Metadata를 확인합니다. 검색 근거가 없으면 Source, Chunk 크기, 질문 표현과 색인 상태를 검토합니다.

## 14. 최종 보안 Checklist

- [ ] Repository가 검증된 `main` Commit인가
- [ ] 실제 Source, 문서, DB, Log와 비밀정보가 외부 Repository에 없는가
- [ ] wheel이 대상 Windows/Python과 호환되는가
- [ ] 모델 사용 권한과 반입 승인이 확인됐는가
- [ ] `SHA256SUMS.txt`가 폐쇄망에서 모두 일치하는가
- [ ] GitHub Credential과 외부 API Key가 반입되지 않았는가
- [ ] `config.local.yaml`이 Git에서 제외되는가
- [ ] `--no-index` 설치와 Offline 환경변수로 검증했는가
- [ ] 실제 Source는 사내 경로에서만 연결되는가
- [ ] 사내 Gitea Backup과 Rollback 단위가 정의됐는가

## 15. 공식 참고 문서

- [Git bundle](https://git-scm.com/docs/git-bundle)
- [pip download](https://pip.pypa.io/en/stable/cli/pip_download/)
- [pip install](https://pip.pypa.io/en/stable/cli/pip_install/)
- [Ollama Modelfile](https://docs.ollama.com/modelfile)
- [Ollama 모델 반입](https://docs.ollama.com/import)
