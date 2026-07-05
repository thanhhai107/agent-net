# Báo cáo đề tài nghiên cứu theo format CVPR

Thư mục này chứa bản báo cáo rút gọn theo template sinh viên báo cáo với mentor, nhưng trình bày bằng format CVPR gốc: `10pt`, `twocolumn`, `letterpaper`, dùng `cvpr.sty`.

- File chính: `main.tex`
- Nội dung các mục: `sec/1_gioi_thieu.tex` đến `sec/5_ket_luan.tex`
- Tài liệu tham khảo: `main.bib`
- PDF đã build: `main.pdf`

Biên dịch:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

PDF hiện tại có 4 trang, dưới giới hạn 6 trang.
