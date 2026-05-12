# 企業多模態 RAG 智慧知識庫

> 基於多查詢融合檢索與自動化品質評估框架之實踐


---

## 專案簡介

本專案實作一套適用於台灣企業環境的多模態 RAG（Retrieval-Augmented Generation）知識庫系統。

系統能同時處理**文字、圖片、音訊**三種格式的企業內部資料，讓員工透過自然語言提問，即可從公司文件、會議記錄、政策圖片中取得有來源根據的精準答案，嚴防 AI 產生幻覺。

---

## 核心特色

- **多查詢 RRF 融合檢索**：透過多路交叉驗證大幅提升召回穩定性
- **真正的多模態支援**：文字 / 圖片 / 音訊統一進入同一檢索流程
- **合規優先設計**：Policy-Aware Boost 確保 HR 合規文件優先顯示
- **自動化品質評核**：LLM-as-a-Judge 全自動評估四大指標
- **繁體中文在地化**：Prompt Engineering 嚴格禁止簡體字輸出

---

## 系統架構

```
使用者提問
    │
    ▼
┌─────────────────────────────────────────────┐
│              Query Expansion                │
│        Llama-3.1 改寫 3 種問法               │
└────────────────┬────────────────────────────┘
                 │  ×3 queries
                 ▼
┌─────────────────────────────────────────────┐
│           Vector Retrieval (FAISS)          │
│         各路各取 Top-10，共 ~30 筆            │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         Reciprocal Rank Fusion (RRF)        │
│     Score = Σ 1 / (k + rank(q, d))         │
└────────────────┬────────────────────────────┘
                 │  Top-20
                 ▼
┌─────────────────────────────────────────────┐
│          Cross-Encoder Reranker             │
│   BGE-reranker-base + Policy-Aware Boost   │
└────────────────┬────────────────────────────┘
                 │  Top-3
                 ▼
┌─────────────────────────────────────────────┐
│         Generation (Llama-3.1-8b)           │
│       僅根據 Context 生成，禁止幻覺           │
└─────────────────────────────────────────────┘
```

---

## 資料預處理

所有格式統一輸出結構：`ID | Text | Modality | Category | Topic | Date`

| 資料類型 | 核心工具 | 處理方式 |
|----------|----------|----------|
| 文字 `.txt` | Regex Parser | 按標題切塊（Heading-based），自動抓取日期存入 Metadata |
| 圖片 `.jpg` | LLaVA-1.6-7b | 4-bit 量化推論，視覺資訊轉繁體中文語意描述 |
| 音訊 `.mp3` | Whisper medium | 語音轉文字 + Sliding Window（300字／50字重疊） |

---

## Model Stack

| 技術階段 | 模型 | 說明 |
|----------|------|------|
| 向量搜尋 | `BAAI/bge-m3` | 多語言 Dense Embedding，中文效果穩定 |
| 精排重排序 | `bge-reranker-base` | Cross-Encoder 聯合編碼機制 |
| 文字生成 | `Llama-3.1-8b` | 透過 Groq API 進行亞秒級高速推論 |
| 視覺理解 | `LLaVA-1.6-7b` | 4-bit 量化，精準圖轉文 |
| 評核模型 | `Llama-4-Scout 17B` | 擔任 LLM-as-a-Judge 評審角色 |

---

## 精排設計

### Cross-Encoder Reranker
使用 `BGE-reranker-base` 同時輸入「問題 + 候選段落」進行深層語意比對，模型真正「讀過」內容再打分，而非單純計算向量空間距離。

### Policy-Aware Boost
偵測到「規定」、「政策」等關鍵字時，對 HR/Policy 類別文件額外加分 **+0.3**，確保企業合規文件優先顯示。

### Dynamic Filtering
支援依日期或模態進行硬性過濾，例如：

```python
# 只搜尋 2023 年之後的會議記錄圖檔
results = retriever.search(
    query="Q3 業績檢討",
    filter={"date": {"gte": "2023-01-01"}, "modality": "image"}
)
```

---

## 評核指標（DeepEval + Llama-4-Scout）

| 指標 | 問題 | 說明 |
|------|------|------|
| **Contextual Recall** | 找全了嗎？ | 正確答案所需資訊是否全數被檢索到 |
| **Contextual Precision** | 找準了嗎？ | 檢索到的段落是否與問題高度相關 |
| **Faithfulness** | 亂說了嗎？ | 回答內容是否完全來自 Context，嚴禁幻覺 |
| **Answer Relevancy** | 答對題了嗎？ | 生成答案是否直接正確回應使用者 |

---

## Ablation Study

| 檢索策略 | 指標 | 結果 |
|----------|------|------|
| Pure Dense（FAISS） | MRR | 70% |
| Dense + BM25 | MRR | 62% ⬇音訊口語噪聲干擾） |
| **RRF + Rerank（最終）** | **Hit Rate** | **100% ** |

> **結論**：加入 BM25 反而使效果下降，因為音訊轉錄含大量口語噪聲，BM25 關鍵字匹配易受干擾。最終採用「多查詢語意融合 + Rerank」達成最優解。

---

## 企業場景實測結果

評核門檻：指標總分 > 0.5 即視為通過

| 測試問題 | 召回分數 | 生成分數 | 結果 |
|----------|----------|----------|------|
| 遠端工作規定與通訊工具 | 1.0 | 1.0 |  PASS |
| 績效考核如何影響薪酬晉升 | 1.0 | 1.0 |  PASS |
| 資安人員安全意識措施 | 1.0 | 1.0 | PASS |
| AI 導入計畫的適用範圍 | 1.0 | 1.0 |  PASS |
| 新考核系統計畫試行時程 | 1.0 | 1.0 | PASS |

**通過率：100%　｜　所有指標全數滿分**

---

## 工程亮點

### 模組化設計（Decoupled Design）
預處理、檢索、生成三個模組完全解耦。新增 PDF 或影片模態時，只需擴充 Ingestion 模組，不影響系統主體。

```
ingestion/
├── text_parser.py      # 文字預處理
├── image_captioner.py  # 圖片轉文字（LLaVA）
└── audio_transcriber.py # 音訊轉文字（Whisper）

retrieval/
├── query_expander.py   # 查詢擴展
├── vector_search.py    # FAISS 向量搜尋
└── rrf_fusion.py       # RRF 排名融合

reranker/
├── cross_encoder.py    # BGE 精排
└── policy_boost.py     # 業務加權邏輯

generation/
└── answer_generator.py # Llama 生成
```

### 資源管理（Resource Management）
- 實作模型防重複載入機制，避免記憶體浪費
- LLaVA-1.6-7b 採用 4-bit 量化，可穩定運行於單張消費級 GPU

### 在地化（Localization）
- 全流程繁體中文優化
- 透過 Prompt Engineering 嚴格禁止簡體字輸出
- 完整適應台灣企業部署環境

---

## 未來演進計畫

### 技術與工程
- [ ] **Prompt 路由**：根據問題類型自動挑選最合適的 Prompt 版本
- [ ] **轉錄修正**：針對 Whisper 輸出加入語意後處理，修正中文同音異字
- [ ] **快取機制**：導入 Semantic Cache，節省重複問題的運算成本

### 業務與深度
- [ ] **多輪對話**：支援上下文記憶，讓使用者能針對特定文件連續追問
- [ ] **測試集擴充**：測試樣本從 5 題擴充至 500+ 題，提升指標說服力
- [ ] **Agentic RAG**：引入自動自我修正機制，檢索失敗時自動調整策略

---

## 專案結構

```
enterprise-multimodal-rag/
├── ingestion/              # 資料預處理模組
├── retrieval/              # 檢索與融合模組
├── reranker/               # 精排模組
├── generation/             # 生成模組
├── evaluation/             # 品質評核模組
├── data/                   # 測試資料
├── configs/                # 設定檔
└── README.md
```

---

## License

MIT License © Enterprise Multimodal AI Solution Group
