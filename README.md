# MTS Daily arXiv

Parser và CLI không phụ thuộc thư viện ngoài để lấy danh sách bài báo từ bảng
Markdown của [MTS_Daily_ArXiv](https://github.com/zezhishao/MTS_Daily_ArXiv).

## Sử dụng

```bash
python -m mts_daily_arxiv.cli --date 2026-09-22 --limit 15
```

Có thể dùng bản Markdown cục bộ hoặc URL khác:

```bash
python -m mts_daily_arxiv.cli --url ./digest.md
```

Thư viện `parse_papers()` giữ nguyên thứ tự nguồn, bỏ dòng không hợp lệ và
loại bài trùng theo URL. Dữ liệu không được suy đoán hoặc bổ sung từ tiêu đề.

## Kiểm thử

```bash
pytest -q
```
