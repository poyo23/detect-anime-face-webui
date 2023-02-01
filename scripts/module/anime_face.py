import cv2
import os
from tqdm import tqdm
import argparse
#import gradio as gr

image_exts = [".png",".jpg",".jpeg"]

class AnimeFaceDetector:
    def __init__(self,
                 padding,enable_padding_ratio,padding_ratio,
                 y_offset, enable_y_offset_ratio, y_offset_ratio,
                 sclae_factor=1.1,min_neighbors=5,detection_output=False):
        # processing arguments
        self.padding=padding
        self.enable_padding_ratio = enable_padding_ratio
        self.padding_ratio = padding_ratio

        self.y_offset = y_offset
        self.enable_y_offset_ratio = enable_y_offset_ratio
        self.y_offset_ratio = y_offset_ratio


        self.detection_output=detection_output # debug option
        self.sclae_factor = sclae_factor
        self.min_neighbors = min_neighbors
        
        extension_dir = os.path.dirname(__file__) # このファイルのディレクトリを参照する。
        # loading cascade
        cascade_file = f"lbpcascade_animeface.xml"
        cascade_path = os.path.join(extension_dir,cascade_file)
        assert os.path.isfile(cascade_path), f"{cascade_path} is not found!"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    
    def __crop(self,cv_img,face_area,padding,offsets):
        max_width = cv_img.shape[1]
        max_height = cv_img.shape[0]
        x, y, w, h = face_area
        x_offset,y_offset = offsets
        bottom,top,right,left = y+h+padding+1,y-padding,x+w+padding+1,x-padding
        # offset
        bottom,top,right,left = bottom+y_offset,top+y_offset,right+x_offset,left+x_offset
        
        bottom = min(bottom,max_height)
        top = max(top,0)
        right = min(right,max_width)
        left = max(left,0)
        crop_area = left,top,right,bottom
        _crop_img = cv_img[top:bottom,left:right]
        return _crop_img,crop_area
    
    def get_offset(self,area):
        x_offset = 0
        y_offset = 0

        if self.enable_y_offset_ratio:
            _,_,w,h = area
            y_offset = int(h*self.y_offset_ratio)
        else:
            y_offset = self.y_offset

        return (x_offset,y_offset)

    def get_padding(self,area):
        if self.enable_padding_ratio:
            _,_,w,h = area
            return int(w*self.padding_ratio)
        else:
            return self.padding


    def detect(self,image_path,output_directory,debug_output_directory):
        filename,fileext = os.path.splitext(os.path.basename(image_path))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = self.sclae_factor,
                                     minNeighbors = self.min_neighbors,
                                     minSize = (24, 24))
        if self.detection_output:
            detect_image = image.copy()
        # each face
        for i,face_area in enumerate(faces):
            _padding = self.get_padding(face_area)
            _offsets = self.get_offset(face_area)
            crop_img,crop_area = self.__crop(image,face_area,_padding,_offsets)
            output_path = os.path.join(output_directory,f"{filename}-{i:02}{fileext}")
            cv2.imwrite(output_path, crop_img)
            if self.detection_output:
                x,y,bottom_x,left_y = crop_area
                cv2.rectangle(detect_image, (x, y), (bottom_x, left_y), (0, 0, 255), 2)
                
        if self.detection_output:
            output_path = os.path.join(debug_output_directory,f"{filename}-rect{fileext}")
            cv2.imwrite(output_path, detect_image)

        return len(faces) != 0





def detect(input_directory,output_directory,debug_output_directory,
           input_recursive,output_recursive,
           padding,enable_padding_ratio,padding_ratio,
           y_offset, enable_y_offset_ratio, y_offset_ratio,
           detection_output,
           sclae_factor,min_neighbors):

    if output_directory == "" or output_directory is None:
        return "Output directory is not specified..."

    if output_recursive and not input_recursive:
        return "To make the output destination recursive, the input recursive must be enabled."
    

    afd = AnimeFaceDetector(padding,enable_padding_ratio,padding_ratio,
                            y_offset, enable_y_offset_ratio, y_offset_ratio,
                            sclae_factor=sclae_factor,min_neighbors=min_neighbors,detection_output=detection_output)

    

    if input_recursive:
        recursive = search_directory(input_directory)

    if output_recursive:
        # 再帰的にディレクトリを作っておく
        for d in recursive.sub_dirs:
            os.makedirs(os.path.join(output_directory,d),exist_ok=True)
    else:
        os.makedirs(output_directory,exist_ok=True)

    if debug_output_directory == None or debug_output_directory == "":
        debug_output_directory = output_directory
    else:
        os.makedirs(debug_output_directory,exist_ok=True)

    undetect_images = []
    if input_recursive:
        for image_path in tqdm(recursive.all_files,desc="Detecting"):
            _image_path = os.path.join(input_directory,image_path)
            if output_recursive:
                _output_dir = os.path.join(output_directory,os.path.dirname(image_path))
                _debug_output_dir = os.path.join(debug_output_directory,os.path.dirname(image_path))
            else:
                _output_dir = output_directory
                _debug_output_dir = debug_output_directory
            _detect = afd.detect(_image_path,_output_dir,_debug_output_dir)
            if not _detect:
                undetect_images.append(image_path)
    else:
        image_path_list = [os.path.join(input_directory,f) for f in os.listdir(input_directory) if any([f.endswith(ext) for ext in image_exts])]
        for image_path in tqdm(image_path_list,desc="Detecting"):
            _detect = afd.detect(image_path,output_directory,debug_output_directory)
            if not _detect:
                undetect_images.append(image_path)
        
    results_text = "Face is not dound in imgaes:<br/>" + "<br/>".join(undetect_images)
    return results_text




def search_directory(input_dir):
    dirs = [""]
    searched_directories = []
    all_files = []
    pbar = tqdm(desc="file searching")
    while len(dirs) > 0:
        d = dirs.pop()
        if ".ipynb_checkpoints" in d:
            continue
        searching_dir = os.path.join(input_dir,d)
        searched_directories.append(d)
        _l = [os.path.join(d,p) for p in os.listdir(searching_dir)]
        dir_list = [p for p in _l if os.path.isdir(os.path.join(input_dir,p))]
        file_list = [p for p in _l if os.path.isfile(os.path.join(input_dir,p))]

        dirs += dir_list
        all_files += file_list
        pbar.update(1)
    pbar.close()

    return argparse.Namespace(files=all_files,sub_dirs=searched_directories)





