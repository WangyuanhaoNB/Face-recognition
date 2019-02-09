#!/usr/local/bin/python3.4 
import flask
import os
import re
import urllib
import hashlib
import collections
from lxml import etree
import os, sys, tempfile, shutil, contextlib
import cv2
import time
import numpy as np

USERNAME='myname'

def make_filename(url,extension):
    if type(url) != bytes:
        url = url.encode('utf8')
    return hashlib.sha1(url).hexdigest() + '.' + extension

def get_image_info(filename):
    img = cv2.imread(filename)
    if img is not None:
        height, width, channels = img.shape
        #FACE_DATA_PATH = '/home/ecegridfs/a/ee364/site-packages/cv2/data/haarcascade_frontalface_default.xml'
        FACE_DATA_PATH = os.path.join(sys.path[0], "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(FACE_DATA_PATH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    else:
        width = 0
        height = 0
        faces = ()
    return {'w':width,'h':height,'faces':faces}

def find_profile_photo_filename(filename_to_etree):
    imgurl = ''
    with fetch_images(filename_to_etree) as filename_to_node:
        for imgfile,imgurl in filename_to_node.items():
            imginfo = get_image_info(imgfile)
            if imginfo['faces'] != ():
                return  imgurl
    return imgurl
    
def add_glasses(filename, face_info):
    MIN_WIDTH=600
    if face_info != ():
        x, y, w, h = face_info[0].tolist()
        if w < MIN_WIDTH :
            img = cv2.imread(filename)
            height, width, channels = img.shape
            img = cv2.resize(img, (MIN_WIDTH, int(height/width*MIN_WIDTH)),interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(filename,img,[int(cv2.IMWRITE_JPEG_QUALITY), 70])
            imginfo = get_image_info(filename)
            face_info = imginfo['faces']
            x, y, w, h = face_info[0].tolist()
            
        else:
            img = cv2.imread(filename)
            height, width, channels = img.shape
        faceimg = img[y:y+h,x:x+w]
        #EYE_DATA = "/home/ecegridfs/a/ee364/site-packages/cv2/data/haarcascade_eye.xml"
        #EYE_DATA = "/home/ecegridfs/a/ee364/site-packages/cv2/data/haarcascade_lefteye_2splits.xml"
        EYE_DATA = os.path.join(sys.path[0], "haarcascade_lefteye_2splits.xml")
        eye_cascade = cv2.CascadeClassifier(EYE_DATA)
        gray = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray',gray)
        #cv2.waitKey(0)        
        eyes1 = eye_cascade.detectMultiScale(gray, 1.3, 5)
        #EYE_DATA = "/home/ecegridfs/a/ee364/site-packages/cv2/data/haarcascade_righteye_2splits.xml"
        EYE_DATA = os.path.join(sys.path[0], "haarcascade_righteye_2splits.xml")
        eye_cascade = cv2.CascadeClassifier(EYE_DATA)
        gray = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray',gray)
        #cv2.waitKey(0)        
        eyes2 = eye_cascade.detectMultiScale(gray, 1.3, 5)
        if eyes1 != () and eyes2 !=() :
            lx, ly, lw, lh = eyes2[0].tolist()
            rx, ry, rw, rh = eyes1[0].tolist()
            x, y, w, h = x + lx, y + ry, rx-lx+max(lw,rw), max(lh,rh)

            cv2.circle(img,(x+h//2,y+h//2), h//2, (100,0,253), 4)
            cv2.circle(img,(x+w-h//2,y+h//2), h//2, (100,0,253), 4)
            cv2.line(img,(x-h//2,y+h//4),(x,y+h//2),(100,0,253), 4)
            cv2.line(img,(x+w+h//2,y+h//4),(x+w,y+h//2),(100,0,253), 4)
            cv2.line(img,(x+h//2+h//2,y+h//2),(x+w-h//2-h//2,y+h//2),(100,0,253), 4)
            #cv2.imshow('rst',img)
            #cv2.waitKey(0)
            img = cv2.resize(img, (width, height),interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(filename,img,[int(cv2.IMWRITE_JPEG_QUALITY), 70])

def copy_profile_photo_to_static(etree):
    base_url = etree.xpath('//base/@url')[0]
    profileUrl = find_profile_photo_filename(etree)
    if profileUrl != '':
        proj_dir = sys.path[0] # e.g., "/home/ecegridfs/a/ee364z15/hpo"
        static_dir = os.path.join(proj_dir, "static") # e.g., "/home/ecegridfs/a/ee364z15/hpo/data"  
        newProfileUrl = profileUrl # //http
        if profileUrl[:2] == '//':
            newProfileUrl = 'http' + profileUrl
        else:
            if profileUrl[:4] != 'http':
                newProfileUrl = base_url + profileUrl
        headers =  {'User-Agent':'PurdueUniversityClassProject/1.0 ({}@purdue.edu https://goo.gl/dk8u5S)'.format(USERNAME)}
        app.logger.info('request for {}'.format(newProfileUrl))
        request = urllib.request.Request(url = newProfileUrl,headers = headers)
        resp= urllib.request.urlopen(request)
        filename = make_filename(newProfileUrl, 'jpg')
        with open(os.path.join(static_dir,filename), "wb") as f:
                f.write(resp.read())
        return profileUrl,filename
    return '',''

@contextlib.contextmanager 
def pushd_temp_dir(base_dir=None, prefix="tmp.hpo."):
    ''' 
    Create a temporary directory starting with {prefix} within {base_dir} 
    and cd to it. 
 
    This is a context manager.  That means it can---and must---be called using 
    the with statement like this: 
 
        with pushd_temp_dir(): 
            ....   # We are now in the temp directory 
        # Back to original directory.  Temp directory has been deleted. 
 
    After the with statement, the temp directory and its contents are deleted. 
 
    Putting the @contextlib.contextmanager decorator just above a function 
    makes it a context manager.  It must be a generator function with one yield.  
 
    - base_dir --- the new temp directory will be created inside {base_dir}. 
                   This defaults to {main_dir}/data ... where {main_dir} is 
                   the directory containing whatever .py file started the 
                   application (e.g., main.py). 
 
    - prefix ----- prefix for the temp directory name.  In case something 
                   happens that prevents 
    ''' 
    if base_dir is None:
        proj_dir = sys.path[0] # e.g., "/home/ecegridfs/a/ee364z15/hpo"
        main_dir = os.path.join(proj_dir, "data") # e.g., "/home/ecegridfs/a/ee364z15/hpo/data"
        # Create temp directory 
    temp_dir_path = tempfile.mkdtemp(prefix=prefix, dir=base_dir) 

    try: 
        start_dir = os.getcwd()  # get current working directory 
        os.chdir(temp_dir_path)  # change to the new temp directory 

        try: 
            yield 
        finally: 
            # No matter what, change back to where you started. 
            os.chdir(start_dir) 
    finally: 
        # No matter what, remove temp dir and contents. 
        shutil.rmtree(temp_dir_path, ignore_errors=True) 

@contextlib.contextmanager   
def fetch_images(etree):
    base_url = etree.xpath('//base/@url')[0]
    with pushd_temp_dir():
        filename_to_node = collections.OrderedDict()
        #
        # Extract the image files into the current directory 
        #
        imgurls = etree.xpath('//img//@src')
        for imgurl in imgurls:
            newimgurl = imgurl # //http
            if imgurl[:2] == '//':
                newimgurl = 'http' + imgurl
            else:
                if imgurl[:4] != 'http':
                    newimgurl = base_url + imgurl
                
            headers =  {'User-Agent':'PurdueUniversityClassProject/1.0 ({}@purdue.edu https://goo.gl/dk8u5S)'.format(USERNAME)}
            request = urllib.request.Request(url = newimgurl,headers = headers)
            resp= urllib.request.urlopen(request)
            filename = make_filename(newimgurl, 'jpg')
            with open(filename, "wb") as f:
                    f.write(resp.read())
            filename_to_node[filename] = imgurl
            app.logger.info('fetch_images {} {}'.format(imgurl,filename))
        yield filename_to_node

app = flask.Flask(__name__) 
 
@app.route('/') 
def root_page(): 
    return flask.render_template('root.html') 

@app.route('/view/') 
def view_page():
    url = flask.request.args.get('url')
    if url:
        urlRegex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        notSocialNetworkRegex = re.compile(r'^((?!facebook|whatsapp|wx.qq|qzone.qq|tumblr|instagram|twitter|plus\.?google|pinterest|skype"linkedin|foursquare|vimeo|tieba.baidu|viber|weibo|line|snapchat|telegram|reddit|youtube|flickr|tiktok).)*$', re.IGNORECASE)
        if urlRegex.match(url) is not None:
            if notSocialNetworkRegex.match(url) is not None:
                headers =  {'User-Agent':'PurdueUniversityClassProject/1.0 ({}@purdue.edu https://goo.gl/dk8u5S)'.format(USERNAME)}
                # Credit: Adapted from example in Python 3.4 Documentation, urllib.request 
                #         License: PSFL https://www.python.org/download/releases/3.4.1/license/ 
                #                  https://docs.python.org/3.4/library/urllib.request.html    
                app.logger.info('request for {}'.format(url))
                request = urllib.request.Request(url = url,headers = headers)
                response = urllib.request.urlopen(request)
                html = response.read().decode('utf-8')
                urlbase=url[:url.find('/',url.find('//')+2)]
                app.logger.info('urlbase: {}'.format(urlbase))
                html = "<base href='{}' url='{}'>{}".format(urlbase,url,html)
                node = etree.HTML(html)
                profileUrl,filename = copy_profile_photo_to_static(node)
                app.logger.info('found profile: {} {}'.format(profileUrl,filename))
                if profileUrl != '':
                    proj_dir = sys.path[0] # e.g., "/home/ecegridfs/a/ee364z15/hpo"
                    static_dir = os.path.join(proj_dir, "static") # e.g., "/home/ecegridfs/a/ee364z15/hpo/data"  
                    abs_filename = os.path.join(static_dir,filename)
                    imginfo = get_image_info(abs_filename)
                    app.logger.info('abs_filename imginfo: {} {}'.format(abs_filename,imginfo))
                    add_glasses(abs_filename,imginfo['faces'])
                    static_url = flask.url_for('static', filename=filename) 
                    app.logger.info('static_url: {}'.format(flask.request.base_url + static_url))
                    #return flask.redirect(static_url)
                    html = html.replace(profileUrl,flask.request.base_url.replace('/view/','') + static_url + '?tm=' + str(time.time()))

                    return html
                return "Profile not found."
            else:
                return "You can not access this web site."
        else:
            return "Error: NOT A WEB SITE URL."
    return flask.render_template('root.html') 
    #123456

if __name__ == '__main__': 
    #filename = '43cdcd7c62213cf0af22cef70f330af53987d490.jpg'
    #proj_dir = sys.path[0] # e.g., "/home/ecegridfs/a/ee364z15/hpo"
    #static_dir = os.path.join(proj_dir, "static") # e.g., "/home/ecegridfs/a/ee364z15/hpo/data"  
    #abs_filename = os.path.join(static_dir,filename)
    #imginfo = get_image_info(abs_filename)
    #add_glasses(abs_filename,imginfo['faces'])

    app.run(host="127.0.0.1", port=os.environ.get("ECE364_HTTP_PORT", 8000), 
            use_reloader=True, use_evalex=False, debug=True, use_debugger=False) 
    # Each student has their own port, which is set in an environment variable. 
    # When not on ecegrid, the port defaults to 8000.  Do not change the host, 
    # use_evalex, and use_debugger parameters.  They are required for security. 
    # 
    # Credit:  Alex Quinn.  Used with permission.  Preceding line only. 
