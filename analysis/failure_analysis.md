# Failure Analysis — Lab 18: Production RAG

**Nhóm:**
**Thành viên:** Đặng Trần Trung Dũng

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---|---|---|
| **Faithfulness** | 0.9469 | 0.9804 | +0.0335 (+3.5%) |
| **Answer Relevancy** | 0.7752 | 0.7869 | +0.0117 (+1.5%) |
| **Context Precision** | 0.9250 | 0.9375 | +0.0125 (+1.4%) |
| **Context Recall** | 0.9250 | 0.8167 | -0.1083 |

---

## Bottom-5 Failures

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Context lấy được thông tin quy trình mua sắm chung nhưng thiếu chunk chứa bảng thẩm quyền hạn mức trên 50 triệu.
- **Worst metric:** Context Recall (0.00)
- **Error Tree:** Output thiếu thông tin chi tiết → Context thiếu chunk thẩm quyền cấp CEO → Retrieval bỏ sót chunk phân cấp mua sắm.
- **Root cause:** Chunking chia nhỏ văn bản quy trình mua sắm khiến bảng phân cấp thẩm quyền (mức > 50tr) bị tách rời khỏi tiêu đề quy trình; Dense Retrieval ưu tiên các chunk nói chung về thiết bị văn phòng.
- **Suggested fix:** Sử dụng Structure-aware chunking hoặc Hierarchical Parent-Child Chunking để giữ toàn vẹn bảng ma trận phân quyền trong cùng một parent chunk.

---

### #2
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Context lấy được thẩm quyền Director phê duyệt (5-50tr) nhưng thiếu chunk quy định xác nhận cấu hình từ phòng CNTT và yêu cầu 3 báo giá.
- **Worst metric:** Context Recall (0.33)
- **Error Tree:** Output trả lời đúng một phần → Context chỉ có thông tin tài chính, thiếu thông tin kỹ thuật CNTT → Query phức tạp đa ý (multi-aspect).
- **Root cause:** Câu hỏi đa khía cạnh (thẩm quyền duyệt chi + quy trình CNTT + thủ tục báo giá) nằm rải rác ở 2-3 phần khác nhau. Reranker top-3 chỉ giữ được chunk có độ tương đồng bề mặt cao nhất với mức giá 30 triệu.
- **Suggested fix:** Áp dụng Query Decomposition (chia nhỏ câu hỏi thành các sub-queries: "Thẩm quyền mua 30tr", "Quy định mua thiết bị CNTT", "Thủ tục báo giá mua sắm") và truy vấn đa luồng trước khi RRF.

---

### #3
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Context tìm được thẩm quyền CEO duyệt khung 16-30 ngày nhưng thiếu chi tiết ràng buộc về nghĩa vụ đóng BHXH khi nghỉ quá 14 ngày.
- **Worst metric:** Context Recall (0.50)
- **Error Tree:** Output đúng người phê duyệt nhưng thiếu điều khoản nghĩa vụ đi kèm → Context bị cắt ngắn trước phần lưu ý bảo hiểm.
- **Root cause:** Kích thước chunk quá nhỏ ngắt quãng ngay sau bảng số ngày nghỉ, làm mất phần ghi chú điều khoản bảo hiểm ở cuối điều khoản.
- **Suggested fix:** Bổ sung Contextual Enrichment (tự động gắn tóm tắt ngữ cảnh toàn bộ quy chế nghỉ phép vào đầu chunk) và tăng sliding window overlap.

---

### #4
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Context chỉ truy xuất được Chính sách an toàn thông tin chung (Cấp 3 - Bí mật) mà thiếu Quy chế chi trả lương.
- **Worst metric:** Context Recall (0.50)
- **Error Tree:** Thiếu thông tin liên kết chéo giữa quy chế nhân sự và chính sách an toàn thông tin.
- **Root cause:** Dữ liệu nằm ở hai tài liệu có ontology khác nhau (một bên dùng thuật ngữ "quy chế lương", một bên dùng "phân loại dữ liệu cấp độ 3").
- **Suggested fix:** Tạo HyQA (Hypothetical Question Generation) trong M5 Enrichment để gắn các câu hỏi liên quan đến phân loại lương vào chunk quy chế lương.

---

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Context lấy được chính sách ngày phép v2024 nhưng thiếu bảng phân bậc thang lương Senior.
- **Worst metric:** Context Recall (0.50)
- **Error Tree:** Multi-hop reasoning cần kết hợp 2 văn bản độc lập (Chính sách ngày phép 2024 và Thang bảng lương vị trí Senior).
- **Root cause:** Reranker ép top_k=3 khiến chunk về thang bảng lương bị loại khỏi danh sách ngữ cảnh cuối cùng.
- **Suggested fix:** Tăng Rerank candidate pool hoặc áp dụng Hybrid Search kết hợp metadata filtering cho các câu hỏi kết hợp đa chính sách.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Đúng một phần (nêu được Giám đốc phòng ban duyệt nhưng thiếu bước xác nhận cấu hình kỹ thuật từ IT và yêu cầu 3 báo giá).
2. **Context đúng?** → Thiếu ngữ cảnh kỹ thuật CNTT (Context Recall = 0.33).
3. **Query rewrite OK?** → Query gốc quá dài và chứa 2 ý độc lập (thẩm quyền mua sắm + quy định kỹ thuật IT).
4. **Fix ở bước:** Retrieval & Query Transformation — Cần bổ sung Query Decomposition / Multi-Query Retrieval để chia nhỏ câu hỏi trước khi tìm kiếm.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Query Decomposition Agent:** Tự động phát hiện câu hỏi phức tạp đa ý để sinh các sub-queries song song.
- **Temporal & Conflict-Resolution Filter:** Tự động lọc phiên bản chính sách cũ (v2023 / v1.0) để ưu tiên chính sách mới nhất (v2024 / v2.0) dựa trên metadata `version` và `effective_date`.
