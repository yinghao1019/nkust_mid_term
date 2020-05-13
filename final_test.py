from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import numpy as np
import cv2
import requests
import urllib.request as req
import keyboard
#自訂義模組
from recon.img_processing import img_filter
from recon.predict import predict_main
driver=webdriver.Chrome()
base_url='https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dwebmge'
def enter_member():
    #定位登入會員位置
    locate_member=driver.find_element_by_link_text('登入')
    locate_member.click()
    time.sleep(2)
    #定位並輸入帳號
    locate_acc=driver.find_element_by_css_selector('input#id')
    locate_acc.send_keys('F108157110@nkust.edu.tw')
    #抓取輸入密碼元素並輸入
    locate_pass=driver.find_element_by_css_selector('input#ps')
    locate_pass.send_keys('z112517z')
#獲取圖片網址
def fetchImg():
    img_element=driver.find_element_by_xpath('//*[@id="random_img_div1"]/img')
    if img_element.is_displayed():
        img_src=img_element.get_attribute('src')
        print(img_src)
        return img_src
    else:
        print('未定位到img元素')
#載入圖片
def url_to_img(img_src,webnum_value):
    headers={
    'Host': 'ndltd.ncl.edu.tw',
    'Cookie': 'webnum={}'.format(webnum_value),
    }
    print("imgurl:", img_src)
    request=req.Request(img_src,headers=headers)
    resp=req.urlopen(request)
    img=np.asarray(bytearray(resp.read()),np.uint8)
    img=cv2.imdecode(img,cv2.IMREAD_GRAYSCALE)
    return img
def main(url=base_url):
    driver.get(url)
    time.sleep(3)
    enter_member()
    time.sleep(1)
    while True:
        #取得當前cookie中的值
        webnum_value=driver.get_cookie('webnum')['value']
        #抓取圖片網址
        img_url=fetchImg()
        #將Url轉換為圖片格式
        vaild_img=url_to_img(img_url,webnum_value)
        #圖片前處理
        vaild_img=img_filter(vaild_img)
        cv2.imshow('test',vaild_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #read model and predict
        predict_arr=predict_main(vaild_img)
        predict_value=''.join([tag for _,tag in predict_arr])
        print('this pic predict value is{}'.format(predict_value))
        time.sleep(6)
        # 輸入驗證碼
        vaild_key=driver.find_element_by_css_selector('input#validinput')
        vaild_key.send_keys(predict_value)
        time.sleep(1)
        #click login button
        click_enter=driver.find_element_by_css_selector('input#button')
        click_enter.click()
        time.sleep(1)
        #檢查是否有alert存在
        check_exist=EC.alert_is_present()(driver)
        if check_exist:
            time.sleep(1)
            driver.switch_to.alert.accept()
            time.sleep(1)
        else:
            print('validate No error!')
            break
if __name__=='__main__':
    main()
    driver.close()
