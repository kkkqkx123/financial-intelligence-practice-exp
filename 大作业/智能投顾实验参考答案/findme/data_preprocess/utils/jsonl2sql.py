import jsonlines
import pymysql
import re
import torch


def reviewdata_insert(db, obj, tablename, tup, error_sentence):
    result = {i: str(obj[i]) for i in tup}
    mid = ""
    per = ""
    select_re = "select * from %s where id = '%s'" % (tablename, result["id"])
    cursor = db.cursor()
    res = cursor.execute(select_re)
    if res > 0:
        return error_sentence
    for field in tup:
        mid += field + ","
        string = result[field].replace("\"", "\'")
        if field in ["body", "title"]:
            string = re.sub('[^A-Za-z0-9\\.]+', ' ', string)
        per += "\"" + string + "\","

    inesrt_re = "insert into {0}".format(tablename) + "(" + mid[:-1] + ") values (" + per[:-1] + ")"
    try:
        cursor = db.cursor()
        cursor.execute(inesrt_re)
        db.commit()
    except:
        error_sentence.append(inesrt_re)
        print(inesrt_re)
        print("==================================================================")
        return error_sentence
    else:
        return error_sentence
    finally:
        return error_sentence


if __name__ == "__main__":  # 起到一个初始化或者调用函数的作用
    db = pymysql.connect(host="localhost", user="root", password="kira920423", database="covid-news", charset='utf8')
    cursor = db.cursor()
    tablename = "aylien_covid_news_data"
    in_file = 'F:\\news\\covid\\aylien_covid_news_data\\aylien_covid_news_data.jsonl'
    error_sentence = []
    error_num = len(error_sentence)
    i = 0
    with jsonlines.open(in_file) as reader:
        for obj in reader:
            if i % 10000 == 0:
                print(i)
            i += 1
            tup = ['id', 'author', 'title', 'body', 'categories', 'characters_count', 'hashtags', 'keywords','language','links','paragraphs_count','published_at','sentences_count','sentiment','social_shares_count','source']
            error_sentence = reviewdata_insert(db, obj, tablename, tup, error_sentence)
            if error_num != len(error_sentence):
                error_num = len(error_sentence)
                torch.save(error_sentence, "error_sentence.pt")

    cursor.close()
    torch.save(error_sentence, "error_sentence.pt")
