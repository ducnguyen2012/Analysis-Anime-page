This is myself note, i will delete it later
- việc cào data khiến nhiều data đã bị trùng: processing đó là loại bỏ các data trùng lặp
- Có nhiều data không liên quan (ví dụ các link đi tới video thì không liên quan: loại bỏ luôn)
- loại hết tất cả data có encode bị sai: UnicodeEncodeError: 'charmap' codec can't encode character '\u25b3' in position 45: character maps to <undefined>


- Tiêu chí làm việc: https://iai-uet.notion.site/Final-project-y-u-c-u-ti-u-ch-ch-m-i-m-64ba519ccc4443c287567686610dd717?pvs=4
- Link đăng ký nhóm: https://docs.google.com/spreadsheets/d/1JUsiEHndDsrZbO5jw9GqpGrntcr-hSAsqI0OkzzMOMY/edit?usp=sharing
- Mình có thể lên kaggle để check xem ae đang làm những model nào với ML


- mới crawl được toàn bộ tên phim và process nó. Phải có link phim + process, khi đó mới crawl tiếp được thông tin từng phim 
- Dự kiến: crawl overview phim, crawl 5 comment nhiều reaction nhất của từng phim
- nhận xét: chỉ khi mà page load xong thì nó mới bắt đầu ấn vào đường link tiếp theo. 

- công việc 8/5/24: vào từng link phim, crawl thông tin về phim ok. Giờ thì lưu thông tin về cmt, thông tin phim từ dạng text
về dạng csv. Lưu ý nó sẽ là: thông tin, thông tin phim rồi tiếp các cột dưới là thông tin. 
về comment, mình sẽ crawl thêm rồi có các trường: tên người cmt, reaction comment, text comment 
- công việc thêm: Crawl hết data, xử lý nó theo đúng trường, viết vào báo cáo (dự tính tầm 10 trang)


- Một số note khi viết phần crawl data:
- Sử dụng các từ ngữ để miêu tả cách bản thân mình crawl data. 


- nhớ cho hết data vào một folder data riêng. 


9/5/2024: dữ liệu cmt người dùng cần chỉ rõ ra là cmt của phim nào!

cần thêm một list gồm tên bộ phim. Như vậy crawl xong hết thông tin cmt mới append vào 1 list đúng bằng độ dài số người cmt. 

