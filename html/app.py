from flask import Flask, render_template
from DataAnalysisModule import TagJBExtractor  # کلاس Scrapy خودت را ایمپورت کن

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('JB.html')

@app.route('/scrape')
def scrape():
    spider = TagJBExtractor ()
    result = spider.run()  # تابع مناسب را اجرا کن
    return f"Scraping done! {result}"  # می‌توانی نتیجه را روی صفحه نمایش دهی

if __name__ == '__main__':
    app.run(debug=True)
