import urllib.request
import time
import json

client_id = "vBsHawg1Wq_qjNvC0reQ" # 개발자센터에서 발급받은 Client ID 값
client_secret = "PxSX89PKi1" # 개발자센터에서 발급받은 Client Secret 값

def translate_text(texts, c_id = client_id, c_pwd = client_secret):
    output_text = []
    for text in texts:
        
        encText = urllib.parse.quote(text)
        data = "source=ja&target=ko&text=" + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", c_id)
        request.add_header("X-Naver-Client-Secret", c_pwd)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode == 200):
            res = json.loads(response.read().decode('utf-8'))
            result = res['message']['result']['translatedText']
            print(result)
            output_text.append(result)
        else:
            print("Error Code:" + rescode)
            print("Try again...")
            time.sleep(1)
            translate_text(text, c_id, c_pwd)
    
    return output_text


if __name__ == "__main__":
    print(translate_text(['テメーみてーな。あからさまな悪役を一撃でぶっ飛ばす。ヒーローに']))