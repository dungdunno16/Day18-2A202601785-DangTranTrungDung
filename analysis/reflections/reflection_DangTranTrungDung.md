# Individual Reflection — Lab 18: Production RAG Pipeline

**Họ và tên:** Đặng Trần Trung Dũng  
**Mã sinh viên:** 2A202601785  
**Module phụ trách / Hoàn thành:** Toàn bộ 5 Modules (M1: Chunking, M2: Hybrid Search, M3: Reranking, M4: Evaluation & Failure Analysis, M5: Enrichment)

---

## Phần 1: Mapping Bài giảng vào Thực tế (Lecture → Code Implementation)

| Khái niệm trong bài giảng | Module | Hàm / Class cụ thể | Quan sát & Nhận thức thực tế |
|---|---|---|---|
| **Semantic & Structure-Aware Chunking** | M1 | `chunk_semantic()`, `chunk_structure_aware()`, `chunk_hierarchical()` | - Fixed-size chunking của Naive Baseline cắt đôi câu và bảng biểu.<br>- `chunk_semantic()` dùng cosine distance (ngưỡng 0.85) giúp phân nhóm theo mạch ý.<br>- `chunk_structure_aware()` giữ nguyên tiêu đề Markdown (`#`, `##`) và metadata `section`, tạo 104 chunks hoàn chỉnh từ 26 tài liệu. |
| **BM25 + Dense Hybrid Search & Fusion** | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | - Dense search (`BAAI/bge-m3`) giỏi hiểu ý nghĩa tổng quát nhưng dễ trượt số liệu/mã quy định cụ thể (vd: "55 triệu", "PVI", "MFA").<br>- BM25 với phân đoạn từ tiếng Việt (`underthesea`) bắt chính xác từ khóa.<br>- RRF ($k=60$) cân bằng điểm số giữa hai không gian tìm kiếm độc lập mà không cần chuẩn hóa scale. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker._load_model()`, `CrossEncoderReranker.rerank()` | - Bi-Encoder (Retriever) đánh giá Query và Document độc lập (nhanh nhưng nông).<br>- Cross-Encoder (`bge-reranker-v2-m3`) thực hiện Full Cross-Attention giữa từng cặp `(query, document)`, giúp lọc từ Top-20 ứng viên xuống Top-3 context cô đọng, nâng **Context Precision từ 0.9250 lên 0.9375**. |
| **RAGAS 4 Core Metrics** | M4 | `evaluate_ragas()`, `failure_analysis()` | - Đánh giá tách biệt 2 trục: **Retrieval** (`context_precision`, `context_recall`) và **Generation** (`faithfulness`, `answer_relevancy`).<br>- Production pipeline đạt **Faithfulness = 0.9804** (tăng +3.5% so với Naive Baseline 0.9469).<br>- Cây chẩn đoán lỗi (Diagnostic Tree) tự động phân loại đúng nguyên nhân lỗi (do chunking, retrieval hay LLM hallucination). |
| **Document Enrichment (Contextual & HyQA)** | M5 | `contextual_prepend()`, `generate_hypothesis_questions()`, `_enrich_single_call()` | - Giải quyết bài toán *"Orphan Chunk"* (chunk bị cô lập mất ngữ cảnh tài liệu gốc).<br>- Kỹ thuật `_enrich_single_call()` gộp tóm tắt, HyQA, ngữ cảnh và metadata vào 1 prompt JSON duy nhất, tối ưu chi phí và độ trễ. |

---

## Phần 2: Khó khăn Kỹ thuật & Cách Giải quyết (Debugging & Engineering)

### 1. Sự cố Quota & Rate Limit khi làm giàu dữ liệu (HTTP 429)
- **Lỗi gặp phải:**  
  `openai.RateLimitError: Error code: 429 - Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day`
- **Nguyên nhân:** Khi làm giàu 104 chunks qua API online miễn phí (Gemini / OpenRouter), số lượng request vượt hạn mức 50 request/ngày của Free tier.
- **Cách giải quyết:**
  - Thiết kế cơ chế **Circuit Breaker** trong `src/m5_enrichment.py`: Khi phát hiện lỗi 429 hoặc cạn quota, hệ thống tự động ngắt kết nối API và kích hoạt **Extractive Fallback Heuristic** (trích xuất câu chủ đề, sinh câu hỏi HyQA bằng regex cấu trúc, gán metadata phân loại tự động).
  - Tốc độ làm giàu 104 chunks tăng vọt từ hàng phút xuống còn **2.3 giây**, bảo đảm pipeline chạy ổn định 100% offline.

### 2. Sự cố Tràn bộ nhớ GPU khi nạp đồng thời Embedder & Cross-Encoder
- **Lỗi gặp phải:**  
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 3.64 GiB...`
- **Nguyên nhân:** Cả hai mô hình lớn `BAAI/bge-m3` (560M params) và `BAAI/bge-reranker-v2-m3` (560M params) cùng nạp lên VRAM card đồ họa 4GB, dẫn đến tràn bộ nhớ khi CrossEncoder khởi tạo.
- **Cách giải quyết:**
  - Bổ sung logic kiểm tra dung lượng VRAM: Nếu GPU < 6GB, CrossEncoder tự động nạp trên CPU (`device="cpu"`).
  - Do Reranker chỉ cần tính toán cho Top-20 ứng viên mỗi query, thời gian xử lý trên CPU cực nhanh (~15ms/query) mà giải quyết dứt điểm rủi ro CUDA OOM.

### 3. Tách từ tiếng Việt cho BM25 (Vietnamese Tokenization)
- **Lỗi gặp phải:** BM25 mặc định split theo khoảng trắng làm các từ ghép tiếng Việt như *"thâm niên"*, *"bảo hiểm"*, *"phê duyệt"* bị rời rạc thành các unigram riêng biệt, làm giảm độ khớp từ khóa chuyên ngành.
- **Cách giải quyết:** Tích hợp `underthesea.word_tokenize(text, format="text")` vào `segment_vietnamese()` để nối từ ghép bằng dấu gạch dưới (`thâm_niên`, `bảo_hiểm_sức_khỏe`, `phê_duyệt`), giúp BM25 phân biệt chính xác thuật ngữ.

---

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  - Thiết kế cơ chế caching cục bộ (Embedding & Enrichment cache vào đĩa) ngay từ đầu để không tốn thời gian và quota khi tinh chỉnh các module sau.
  - Tích hợp metadata pre-filtering theo phiên bản (`version = v2024`, `status = active`) ngay ở tầng Retriever để loại bỏ triệt để các chunk chính sách cũ đã hết hiệu lực (v2023, v1.0).
  - Tối ưu hóa batch size và tự động điều phối thiết bị (GPU/CPU dynamic scheduling) cho CrossEncoder để đạt throughput cao nhất.
- **Module nào muốn thử tiếp:**
  - **Module 2 (Query Transformation & HyDE):** Thử nghiệm thêm Query Decomposition (phân rã câu hỏi đa ý) và Hypothetical Document Embeddings (HyDE) để cải thiện triệt để `context_recall` cho các câu hỏi phức tạp đa khía cạnh.
  - **Module 5 (Graph & Agentic Enrichment):** Thử nghiệm tích hợp GraphRAG / Entity Linking để kết nối tự động các thực thể ràng buộc giữa nhiều quy chế khác nhau trong doanh nghiệp.

---

## 5. Tự Đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú minh chứng |
|---|:---:|---|
| **Hiểu bài giảng** | **5 / 5** | Nắm vững toàn bộ 5 kỹ thuật: Chunking, Hybrid Search, RRF, Cross-Encoder Reranking, RAGAS Metrics, Contextual Enrichment. |
| **Code quality** | **5 / 5** | Cấu trúc code sạch, đầy đủ type annotations, xử lý triệt để ngoại lệ, hỗ trợ fallback offline thông minh. Đạt **37/37 unit tests** (`100% pass`). |
| **Teamwork / Thực nghiệm** | **5 / 5** | Hoàn thành phân tích chi tiết Top-5 failures theo cây chẩn đoán (Diagnostic Tree) và đo lường định lượng so sánh với Baseline. |
| **Problem solving** | **5 / 5** | Xử lý triệt để các bài toán khó thực tế: CUDA OOM trên GPU 4GB, API Rate Limit (429), Vietnamese Word Segmentation. |
